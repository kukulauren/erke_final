from collections import defaultdict
from app.helper_functions import get_distance, get_box_iou, boxes_overlap
from app.variables import CUSTOMER_DWELL_TIME, SCANNER_PHONE_DISTANCE, SCANNER_ITEM_DISTANCE, SCANNER_MOVEMENT_THRESHOLD, PAYMENT_COMPLETE_TIME, SCAN_COOLDOWN

class RetailAnalytics:
    def __init__(self, fps=30):
        self.fps = fps

        # Scanner tracking
        self.scanner_positions = {}  # scanner_id: [last_positions]
        self.scanner_moving = {}     # scanner_id: bool

        # Item scanning
        self.scanned_items = []
        self.last_scan_time = {}     # item_id: last_scan_time
        self.current_overlaps = []

        # Payment
        self.payment_in_progress = None
        self.completed_payments = []
        self.phone_tracks = defaultdict(list)
        self.payment_times = []
        self.cash_detected=[]

        # Customer
        self.customers_at_counter = {}
        self.customer_visits = []
        self.service_times = []

    def update_scanner_movement(self, scanners, current_time):
        """Track if scanner is moving"""
        scanner_status = {}

        for scanner in scanners:
            scanner_id = scanner.get('track_id') or 0
            current_center = scanner['center']

            if scanner_id not in self.scanner_positions:
                self.scanner_positions[scanner_id] = [current_center]
                scanner_status[scanner_id] = {'moving': False, 'speed': 0}
            else:
                prev_center = self.scanner_positions[scanner_id][-1]
                movement = get_distance(current_center, prev_center)

                is_moving = movement > SCANNER_MOVEMENT_THRESHOLD
                self.scanner_moving[scanner_id] = is_moving

                scanner_status[scanner_id] = {
                    'moving': is_moving,
                    'speed': movement,
                    'center': current_center
                }

                # Keep last 5 positions
                self.scanner_positions[scanner_id].append(current_center)
                if len(self.scanner_positions[scanner_id]) > 5:
                    self.scanner_positions[scanner_id].pop(0)

        return scanner_status

    def update_item_scanning(self, items, scanners, scanner_status, current_time):
        """
        NEW LOGIC: Scanner MOVES and overlaps/near Item = Item Scanned

        Conditions for scan:
        1. Scanner bbox overlaps Item bbox OR distance < threshold
        2. Scanner was MOVING (approaching the item)
        3. Cooldown passed for this item
        """
        events = []
        self.current_overlaps = []

        for scanner in scanners:
            scanner_id = scanner.get('track_id') or 0
            status = scanner_status.get(scanner_id, {})
            is_moving = status.get('moving', False)

            scanner_box = scanner['box']
            scanner_center = scanner['center']

            for item in items:
                item_id = item.get('track_id') or id(item)
                item_box = item['box']
                item_center = item['center']

                # Check overlap (bbox intersection)
                has_overlap = boxes_overlap(scanner_box, item_box)
                iou = get_box_iou(scanner_box, item_box) if has_overlap else 0

                # Also check distance
                dist = get_distance(scanner_center, item_center)
                is_close = dist < SCANNER_ITEM_DISTANCE

                # Either overlap OR very close
                is_scanning_position = has_overlap or is_close

                self.current_overlaps.append({
                    'item_id': item_id,
                    'scanner_id': scanner_id,
                    'distance': dist,
                    'has_overlap': has_overlap,
                    'iou': iou,
                    'is_close': is_close,
                    'scanner_moving': is_moving
                })

                # SCAN DETECTION: Scanner moved and is now overlapping/close to item
                if is_scanning_position and is_moving:
                    # Check cooldown
                    last_scan = self.last_scan_time.get(item_id, 0)
                    if current_time - last_scan > SCAN_COOLDOWN:
                        # SCANNED!
                        self.scanned_items.append({
                            'time': current_time,
                            'item_id': item_id,
                            'scanner_id': scanner_id,
                            'distance': dist,
                            'iou': iou
                        })
                        self.last_scan_time[item_id] = current_time
                        events.append(f"✓ ITEM #{item_id} SCANNED!")
        return events

    def update_payment_scanning(self, phones, scanners, current_time):
        """Phone near scanner for 1s = payment complete (working well)"""
        events = []
        phone_near_scanner = False

        for phone in phones:
            phone_id = phone.get('track_id') or id(phone)

            self.phone_tracks[phone_id].append({
                'position': phone['center'],
                'time': current_time
            })
            if len(self.phone_tracks[phone_id]) > 60:
                self.phone_tracks[phone_id].pop(0)

            for scanner in scanners:
                dist = get_distance(phone['center'], scanner['center'])

                if dist < SCANNER_PHONE_DISTANCE:
                    phone_near_scanner = True

                    if self.payment_in_progress is None:
                        self.payment_in_progress = {
                            'start_time': current_time,
                            'phone_id': phone_id
                        }
                        events.append("📱 PAYMENT STARTED...")
                    else:
                        duration = current_time - self.payment_in_progress['start_time']
                        if duration >= PAYMENT_COMPLETE_TIME:
                            self.completed_payments.append({
                                'time': current_time,
                                'phone_id': phone_id,
                                'duration': duration
                            })
                            self.payment_times.append(duration)
                            events.append("✓ PAYMENT COMPLETE (Mobile)")
                            self.payment_in_progress = None
                    break

        if not phone_near_scanner and self.payment_in_progress:
            self.payment_in_progress = None

        return events

    def detect_cash(self, cashes, customers,current_time):
        events = []
        active_ids = set()

        for cash in cashes:
            cash_id = cash.get('track_id') or id(cash)
            active_ids.add(cash_id)

            for customer in customers:
                if boxes_overlap(cash['box'], customer['box']):
                    customer.setdefault('paid_with_cash', []).append(cash_id)
                    self.cash_detected.append({
                        'event': 'cash_detected',
                        'customer_id': customer.get('track_id') or id(customer),
                        'cash_id': cash_id
                    })
                    break
        return events

    def update_customer_at_counter(self, customers, counters, current_time):
        """Customer bbox overlapping counter"""
        events = []
        active_ids = set()

        for customer in customers:
            customer_id = customer.get('track_id') or id(customer)
            active_ids.add(customer_id)

            for counter in counters:
                has_overlap = boxes_overlap(customer['box'], counter['box'])

                if has_overlap:
                    if customer_id not in self.customers_at_counter:
                        self.customers_at_counter[customer_id] = {
                            'arrival_time': current_time,
                            'counted': False
                        }
                        events.append(f"👤 CUSTOMER #{customer_id} AT COUNTER")
                    else:
                        dwell = current_time - self.customers_at_counter[customer_id]['arrival_time']
                        if dwell >= CUSTOMER_DWELL_TIME and not self.customers_at_counter[customer_id]['counted']:
                            self.customers_at_counter[customer_id]['counted'] = True
                            self.customer_visits.append({'customer_id': customer_id})
                    break

        for cid in list(self.customers_at_counter.keys()):
            if cid not in active_ids:
                dwell = current_time - self.customers_at_counter[cid]['arrival_time']
                self.service_times.append(dwell)
                events.append(f"👤 CUSTOMER #{cid} LEFT ({dwell:.1f}s)")
                del self.customers_at_counter[cid]

        return events

    def get_display_stats(self):
        return [
            f"Items Scanned: {len(self.scanned_items)+1}",
            f"Payments: {len(self.completed_payments)}",
            f"At Counter: {len(self.customers_at_counter)}"
        ]