from collections import defaultdict
from app.helper_functions import get_distance, get_box_iou, boxes_overlap
from app.variables import (
    CUSTOMER_DWELL_TIME,
    SCANNER_PHONE_DISTANCE,
    SCANNER_ITEM_DISTANCE,
    SCANNER_MOVEMENT_THRESHOLD,
    PAYMENT_COMPLETE_TIME,
    SCAN_COOLDOWN,
    STAFF_REENTRY_THRESHOLD,
    STAFF_CUMULATIVE_TIME,
    REENTRY_WINDOW,
    STAFF_CONFIDENCE_THRESHOLD,
    RECENT_BEHAVIOR_WINDOW,
    CONFIDENCE_DECAY_RATE,
    REENTRY_MIN_SESSIONS,
    STAFF_TIME_MIN_SESSIONS
)

class RetailAnalytics:
    # Configuration constants for memory management
    MAX_SCANNER_POSITIONS = 5
    MAX_PHONE_TRACKS = 60
    MAX_SCANNED_ITEMS = 500
    MAX_COMPLETED_PAYMENTS = 500
    MAX_CASH_DETECTED = 500
    MAX_SERVICE_TIMES = 500
    MAX_LAST_SCAN_TIME_ENTRIES = 500
    
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
        self.cash_detected = []

        # Customer
        self.customers_at_counter = {}
        self.customer_visits = []
        self.service_times = []

        # Per-person behavior tracking for staff/customer differentiation
        # Keyed by track_id
        self.person_records = {}  # track_id: {first_seen, last_seen, cumulative_time, in_view, enter_count, last_exit, reentries, reentry_times, classification, staff_source, staff_confidence}
        self.MAX_PERSON_RECORDS = 2000

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

                # Keep last N positions for memory efficiency
                self.scanner_positions[scanner_id].append(current_center)
                if len(self.scanner_positions[scanner_id]) > self.MAX_SCANNER_POSITIONS:
                    self.scanner_positions[scanner_id].pop(0)

        return scanner_status

    def update_person_behavior(self, customers, current_time, cashiers=None):
        """Track per-person on-screen sessions with improved staff/customer differentiation.
        
        IMPROVEMENTS:
        - Confidence-based secondary staff classification (0-1 score)
        - Better re-entry tracking: only counts recent re-entries within a window
        - Minimum thresholds: need multiple sessions to qualify as staff
        - Optimized for single-day operation (no persistence across restarts)
        
        customers: list of detection dicts with 'track_id' and 'center'
        cashiers: list of cashier detection dicts with 'track_id' (primary staff indicator)
        """
        events = []
        active_ids = set()
        cashier_ids = set()

        # FIRST PASS: Mark all detected cashiers as staff (primary indicator)
        if cashiers is None:
            cashiers = []
        
        for cashier in cashiers:
            cid = cashier.get('track_id') or id(cashier)
            cashier_ids.add(cid)
            active_ids.add(cid)

            rec = self.person_records.get(cid)
            if rec is None:
                # first time seeing this cashier
                rec = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'cumulative_time': 0.0,
                    'in_view': True,
                    'enter_count': 1,
                    'last_exit': None,
                    'reentries': 0,
                    'reentry_times': [],
                    'classification': 'staff',
                    'staff_source': 'primary',
                    'staff_confidence': 1.0
                }
                rec['_session_start'] = current_time
                self.person_records[cid] = rec
                events.append(f"👷 PRIMARY STAFF: CASHIER DETECTED #{cid}")
            else:
                # Already-seen cashier still in view
                if not rec.get('in_view'):
                    rec['in_view'] = True
                    rec['enter_count'] = rec.get('enter_count', 0) + 1
                    rec['_session_start'] = current_time
                    rec['staff_confidence'] = 1.0
                
                rec['last_seen'] = current_time
                # Ensure classification is staff (primary staff cannot be downgraded)
                if rec.get('classification') != 'staff' or rec.get('staff_source') != 'primary':
                    rec['classification'] = 'staff'
                    rec['staff_source'] = 'primary'
                    rec['staff_confidence'] = 1.0
                    events.append(f"👷 PRIMARY STAFF: CASHIER CONFIRMED #{cid}")

        # SECOND PASS: Track customers & evaluate for secondary staff classification
        for cust in customers:
            cid = cust.get('track_id') or id(cust)
            active_ids.add(cid)

            # Skip if already marked as primary staff cashier
            if cid in cashier_ids:
                continue

            rec = self.person_records.get(cid)
            if rec is None:
                # First time seeing this person as customer
                rec = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'cumulative_time': 0.0,
                    'in_view': True,
                    'enter_count': 1,
                    'last_exit': None,
                    'reentries': 0,
                    'reentry_times': [],
                    'classification': 'customer',
                    'staff_source': None,
                    'staff_confidence': 0.0
                }
                rec['_session_start'] = current_time
                self.person_records[cid] = rec
            else:
                if not rec.get('in_view'):
                    # Person re-entered after being out of view
                    rec['in_view'] = True
                    rec['enter_count'] = rec.get('enter_count', 0) + 1
                    
                    # Track re-entry time (filter to recent window)
                    if rec.get('last_exit') is not None:
                        rec['reentry_times'] = rec.get('reentry_times', [])
                        rec['reentry_times'].append(current_time)
                        
                        # Only keep recent re-entries within REENTRY_WINDOW
                        rec['reentry_times'] = [t for t in rec['reentry_times'] 
                                               if current_time - t <= REENTRY_WINDOW]
                        
                        # Update re-entry count based on recent window
                        rec['reentries'] = len(rec['reentry_times'])
                    
                    # Start new session
                    rec['_session_start'] = current_time
                
                # Update last seen while in view
                rec['last_seen'] = current_time

            # SECONDARY STAFF EVALUATION: Use confidence-based approach
            if rec.get('classification') != 'staff' or rec.get('staff_source') == 'secondary':
                confidence = self._calculate_staff_confidence(rec, current_time)
                rec['staff_confidence'] = confidence
                
                # Classify based on confidence threshold
                is_confident_staff = confidence >= STAFF_CONFIDENCE_THRESHOLD
                is_currently_staff = rec.get('classification') == 'staff'
                
                if is_confident_staff and not is_currently_staff:
                    # Promote to secondary staff
                    rec['classification'] = 'staff'
                    rec['staff_source'] = 'secondary'
                    events.append(f"👷 SECONDARY STAFF: #{cid} (confidence: {confidence:.2f})")
                
                elif not is_confident_staff and is_currently_staff and rec.get('staff_source') == 'secondary':
                    # Demote from secondary staff
                    rec['classification'] = 'customer'
                    rec['staff_source'] = None
                    rec['staff_confidence'] = 0.0
                    events.append(f"👤 CUSTOMER: #{cid} (confidence: {confidence:.2f})")

        # THIRD PASS: Handle exits
        for cid, rec in list(self.person_records.items()):
            if rec.get('in_view') and cid not in active_ids:
                # Person left the view
                rec['in_view'] = False
                rec['last_exit'] = current_time
                
                # Calculate this session's duration
                start = rec.get('_session_start', rec.get('first_seen', current_time))
                session_time = rec.get('last_seen', current_time) - start
                
                if session_time > 0:
                    rec['cumulative_time'] = rec.get('cumulative_time', 0.0) + session_time
                
                rec.pop('_session_start', None)
                
                # Final secondary staff check at exit
                if rec.get('classification') != 'staff' or rec.get('staff_source') == 'secondary':
                    confidence = self._calculate_staff_confidence(rec, current_time)
                    rec['staff_confidence'] = confidence
                    
                    if confidence >= STAFF_CONFIDENCE_THRESHOLD and rec.get('classification') != 'staff':
                        rec['classification'] = 'staff'
                        rec['staff_source'] = 'secondary'
                        events.append(f"👷 SECONDARY STAFF: #{cid} confirmed on exit (confidence: {confidence:.2f})")

        # Memory management
        if len(self.person_records) > self.MAX_PERSON_RECORDS:
            sorted_items = sorted(self.person_records.items(), key=lambda kv: kv[1].get('last_seen', 0))
            for k, _ in sorted_items[: len(self.person_records) - self.MAX_PERSON_RECORDS]:
                del self.person_records[k]

        return events

    def _calculate_staff_confidence(self, person_rec, current_time):
        """
        Calculate confidence score (0-1) that a person is staff based on behavioral patterns.
        Single-day focused: recent re-entries and cumulative time within current session.
        
        Factors:
        - Recent re-entries (multiple quick returns suggest staff restocking/breaks)
        - Cumulative time across multiple sessions (staff often spend longer)
        """
        confidence = 0.0
        
        enter_count = person_rec.get('enter_count', 0)
        reentries = person_rec.get('reentries', 0)
        cumulative_time = person_rec.get('cumulative_time', 0.0)
        reentry_times = person_rec.get('reentry_times', [])
        
        # Factor 1: Recent re-entry pattern (4+ entries within 10-min window = strong staff indicator)
        recent_reentries = len([t for t in reentry_times if t])
        if recent_reentries >= STAFF_REENTRY_THRESHOLD and enter_count >= REENTRY_MIN_SESSIONS:
            # Scale: 4 reentries = 0.5 confidence, 6+ = higher
            reentry_score = min(0.8, (recent_reentries / STAFF_REENTRY_THRESHOLD) * 0.6)
            confidence += reentry_score
        
        # Factor 2: Cumulative time (15+ minutes across 3+ sessions = moderate staff indicator)
        if enter_count >= STAFF_TIME_MIN_SESSIONS and cumulative_time >= STAFF_CUMULATIVE_TIME:
            # Scale: at threshold = 0.4, double threshold = 0.7
            time_score = min(0.8, (cumulative_time / (STAFF_CUMULATIVE_TIME * 2.0)) * 0.8)
            confidence += time_score
        
        # Cap at 1.0
        return min(1.0, confidence)

    def get_person_label(self, track_id):
        """Get display label for person: shows classification, source, and confidence for secondary staff"""
        rec = self.person_records.get(track_id)
        if not rec:
            return 'unknown'
        
        classification = rec.get('classification', 'unknown')
        source = rec.get('staff_source')
        confidence = rec.get('staff_confidence', 0.0)
        
        if classification == 'staff':
            if source == 'primary':
                return f"{classification}-{source}"
            else:
                # Secondary staff: show confidence score (0-1)
                return f"{classification}-2nd({confidence:.2f})"
        return classification

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
                        # Clean up old scan history to prevent memory growth
                        if len(self.scanned_items) > self.MAX_SCANNED_ITEMS:
                            self.scanned_items.pop(0)
                        
                        self.last_scan_time[item_id] = current_time
                        # Clean up old scan time entries
                        if len(self.last_scan_time) > self.MAX_LAST_SCAN_TIME_ENTRIES:
                            oldest_key = min(self.last_scan_time.keys(), key=lambda k: self.last_scan_time[k])
                            del self.last_scan_time[oldest_key]
                        
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
            if len(self.phone_tracks[phone_id]) > self.MAX_PHONE_TRACKS:
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
                            # Clean up old payment history
                            if len(self.completed_payments) > self.MAX_COMPLETED_PAYMENTS:
                                self.completed_payments.pop(0)
                            
                            self.payment_times.append(duration)
                            events.append("✓ PAYMENT COMPLETE (Mobile)")
                            self.payment_in_progress = None
                    break

        if not phone_near_scanner and self.payment_in_progress:
            self.payment_in_progress = None

        return events

    def detect_cash(self, cashes, customers, current_time):
        """Detect cash near customers and generate events"""
        events = []
        active_ids = set()

        for cash in cashes:
            cash_id = cash.get('track_id') or id(cash)
            active_ids.add(cash_id)

            for customer in customers:
                if boxes_overlap(cash['box'], customer['box']):
                    customer_id = customer.get('track_id') or id(customer)
                    customer.setdefault('paid_with_cash', []).append(cash_id)
                    
                    self.cash_detected.append({
                        'event': 'cash_detected',
                        'customer_id': customer_id,
                        'cash_id': cash_id,
                        'time': current_time
                    })
                    # Clean up old cash detection history
                    if len(self.cash_detected) > self.MAX_CASH_DETECTED:
                        self.cash_detected.pop(0)
                    
                    events.append(f"💵 CASH DETECTED (Customer #{customer_id})")
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
                # Clean up old service times
                if len(self.service_times) > self.MAX_SERVICE_TIMES:
                    self.service_times.pop(0)
                events.append(f"👤 CUSTOMER #{cid} LEFT ({dwell:.1f}s)")
                del self.customers_at_counter[cid]

        return events

    def get_display_stats(self):
        return [
            f"Items Scanned: {len(self.scanned_items)}",
            f"Payments: {len(self.completed_payments)}",
            f"At Counter: {len(self.customers_at_counter)}"
        ]