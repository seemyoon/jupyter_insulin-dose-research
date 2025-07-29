from collections import defaultdict
from datetime import datetime, timedelta

from utils.enums.therapy_type import TherapyType


class MakeWindows:

    @staticmethod
    def to_dt(value):
        try:
            return datetime(value.year, value.month, value.day, value.hour, value.minute)
        except AttributeError:
            return None

    def group_by(self, records):
        """
    before group_by:

    [
     Measurement(patient_id="p1", year=2023, month=1, day=1, hour=9, minute=0, ...),
     Measurement(patient_id="p1", year=2023, month=1, day=1, hour=10, minute=0, ...),
     Measurement(patient_id="p2", year=2023, month=1, day=1, hour=8, minute=30, ...),
    ]

     after group_by:

     {
         "p1": [Measurement(...9:00), Measurement(...10:00)],
         "p2": [Measurement(...8:30)],
     }
        """
        dict_grouped = defaultdict(list)  # By default, the value for the new key will be an empty list [].

        for record in records:
            dict_grouped[record.patient_id].append(record)

        for pat_id in dict_grouped:
            dict_grouped[pat_id].sort(key=self.to_dt)

        return dict_grouped

    def build_feature_windows(self, insulin_recs, drug_tablets_recs, meas_recs, food_intake_recs, window_h=24):

        insulin_by = self.group_by(insulin_recs)
        drug_tablets_by = self.group_by(drug_tablets_recs)
        meas_by = self.group_by(meas_recs)
        food_intake_by = self.group_by(food_intake_recs)

        windows = []

        pat_ids = set(insulin_by) | set(drug_tablets_by) | set(meas_by) | set(
            food_intake_by)  # set of all patient_ids that have at least one record

        for pat_id in pat_ids:
            times = [self.to_dt(r) for r in
                     insulin_by.get(pat_id, []) + drug_tablets_by.get(pat_id, []) + meas_by.get(pat_id,
                                                                                                []) + food_intake_by.get(
                         pat_id, [])]
            """
            EXAMPLE: times = [
                datetime.datetime(2025, 7, 25, 8, 0),    # insulin
                datetime.datetime(2025, 7, 25, 18, 30),  # insulin
                datetime.datetime(2025, 7, 25, 7, 45),   # CGM measurement
                datetime.datetime(2025, 7, 25, 19, 0),   # CBG measurement
                datetime.datetime(2025, 7, 25, 12, 0),   # meal
                datetime.datetime(2025, 7, 25, 8, 30),   # pills
                datetime.datetime(2025, 7, 25, 20, 0),   # pills
            ]
            """

            if not times:
                continue

            start, end = min(times), max(times)
            current_period = start

            while current_period < end:
                next_period = current_period + timedelta(hours=window_h)

                insulin_w = self._filter_by_time(insulin_by.get(pat_id, []), current_period, next_period)

                # insulin_w = [
                #     {'timestamp': '2024-01-02 10:00:00', 'dose': 4.0, 'drug_id': 1},
                #     {'timestamp': '2024-01-04 07:00:00', 'dose': 8.0, 'drug_id': 2}
                # ]

                drug_tablets_w = self._filter_by_time(drug_tablets_by.get(pat_id, []), current_period, next_period)
                meas_w = self._filter_by_time(meas_by.get(pat_id, []), current_period, next_period)
                food_intake_w = self._filter_by_time(food_intake_by.get(pat_id, []), current_period, next_period)

                if not insulin_w and not drug_tablets_w and not meas_w and not food_intake_w:
                    current_period = next_period
                    continue

                insulin_doses_by_type = self._sum_doses_by_type(insulin_w, 'dose', 'insulin_type')
                drug_tablets_by_type = self._sum_doses_by_type(drug_tablets_w, 'dose', 'diabetes_tablet')

                cgm_values = [record.cgm for record in meas_w if record.cgm is not None]
                cbg_values = [record.cbg for record in meas_w if record.cbg is not None]
                blood_ketones = [record.blood_ketone for record in meas_w if record.blood_ketone is not None]

                food_intake_count = len(food_intake_w)

                therapy_type = None
                # combined - 0; insulin - 1; tablet - 2

                if insulin_doses_by_type and drug_tablets_by_type:
                    therapy_type = TherapyType.COMBINED
                elif insulin_doses_by_type:
                    therapy_type = TherapyType.INSULIN
                elif drug_tablets_by_type:
                    therapy_type = TherapyType.TABLET

                windows.append({
                    'patient_id': pat_id,
                    'window_start': current_period,
                    'window_end': next_period,
                    'insulin_doses_by_type': insulin_doses_by_type or None,
                    'drug_tablets_by_type': drug_tablets_by_type or None,
                    'cgm_values': cgm_values,
                    'cbg_values': cbg_values,
                    'blood_ketones': blood_ketones,
                    'food_intake_count': food_intake_count,
                    'therapy_type': therapy_type,
                })

                current_period = next_period

        return windows

    def _filter_by_time(self, records, current_period, next_period):
        return [record for record in records if current_period <= self.to_dt(record) < next_period]

    @staticmethod
    def _sum_doses_by_type(records, dose_attr, related_name_attr) -> dict:
        """

        :param records: list of records (e.g., TakingInsulin).
        :param dose_attr: name of the field with the dose (‘dose’).
        :param related_name_attr: name of the related field pointing to the table with the name of the drug (‘insulin’, ‘diabetes_tablet’, etc.).
        return: example { 1: 14.0, 3: 6.0 }

        """
        dict_grouped = defaultdict(float)

        """
            1. We take all records for insulin (or pills) within the current time window.
            2. For each drug, we extract the med_id (insulin or pill ID).
            3. We take the dose (record.dose) and add it to the counter for that ID.
        """

        for record in records:
            med = getattr(record, related_name_attr, None)
            med_id = getattr(med, 'id', None)

            if med_id:
                dict_grouped[med_id] += getattr(record, dose_attr, 0.0)

        return dict(dict_grouped)
