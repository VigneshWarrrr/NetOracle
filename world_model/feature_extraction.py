"""
NetOracle CIC-IDS-2018 Timestamp-Based Preprocessing

Each network state = one REAL 10-second interval.

21 traffic features
x [mean, std, max, min]
+ flow count
= 85-dimensional state

20 historical states = 200 seconds observed
5 future states      = 50 seconds forecast
"""

import os
import re

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import StandardScaler


class FeatureExtractor:

    RAW_FEATURES = [
        "Dst Port",
        "Protocol",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "SYN Flag Cnt",
        "ACK Flag Cnt",
        "FIN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "URG Flag Cnt",
        "Pkt Size Avg",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
    ]

    def __init__(self, config_path="configs/config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.scaler = StandardScaler()
        self.scaler_fitted = False

        self.label_to_stage = {
            "benign": 0,

            "ftp-bruteforce": 2,
            "ssh-bruteforce": 2,
            "brute force -web": 2,
            "brute force -xss": 2,
            "sql injection": 2,

            "dos attacks-hulk": 3,
            "dos attacks-slowhttptest": 3,
            "dos attacks-goldeneye": 3,
            "dos attacks-slowloris": 3,
            "ddos attack-hoic": 3,
            "ddos attack-loic-udp": 3,

            "infilteration": 4,
            "infiltration": 4,
        }

    # =====================================================
    # COLUMN CLEANING
    # =====================================================

    @staticmethod
    def canonical_column(name):

        name = re.sub(
            r"\s+",
            " ",
            str(name).strip(),
        )

        aliases = {
            "FlowIAT Mean": "Flow IAT Mean",
            "Flow IATMean": "Flow IAT Mean",

            "FIN FlagCnt": "FIN Flag Cnt",
            "SYN FlagCnt": "SYN Flag Cnt",
            "ACK FlagCnt": "ACK Flag Cnt",
            "RST FlagCnt": "RST Flag Cnt",
            "PSH FlagCnt": "PSH Flag Cnt",
            "URG FlagCnt": "URG Flag Cnt",
        }

        return aliases.get(name, name)

    @staticmethod
    def canonical_label(label):

        return re.sub(
            r"\s+",
            " ",
            str(label).strip(),
        ).lower()

    # =====================================================
    # LOAD ONE CSV
    # =====================================================

    def load_file(
        self,
        file_path,
        start_time=None,
        end_time=None,
    ):

        print(
            "\n[*] Loading:",
            os.path.basename(file_path),
        )

        columns = (
            ["Timestamp", "Label"]
            + self.RAW_FEATURES
        )

        chunks = []

        for chunk in pd.read_csv(
            file_path,
            chunksize=100_000,
            low_memory=False,
        ):

            chunk.columns = [
                self.canonical_column(c)
                for c in chunk.columns
            ]

            # Remove repeated header rows.
            chunk = chunk[
                chunk["Label"]
                .astype(str)
                .str.strip()
                != "Label"
            ].copy()

            # Timestamp parsing.
                        # CIC-IDS-2018 timestamps are strings such as:
            # 14/02/2018 08:31:01
            # Force string parsing so numeric-looking values
            # cannot accidentally become Unix timestamps (1970).

            timestamp_text = (
                chunk["Timestamp"]
                .astype(str)
                .str.strip()
            )

            chunk["Timestamp"] = pd.to_datetime(
                timestamp_text,
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce",
            )

            # Safety guard: this dataset is from 2018.
            # Reject malformed timestamps instead of allowing
            # them to create enormous fake timelines.
            valid_year = (
                chunk["Timestamp"].dt.year == 2018
            )

            chunk = chunk.loc[
                valid_year
            ].copy()

            chunk = chunk.dropna(
                subset=["Timestamp"]
            )

            # Optional time filtering.
            if start_time is not None:

                chunk = chunk[
                    chunk["Timestamp"]
                    >= pd.Timestamp(start_time)
                ]

            if end_time is not None:

                chunk = chunk[
                    chunk["Timestamp"]
                    < pd.Timestamp(end_time)
                ]

            if len(chunk) == 0:
                continue

            # Validate features.
            missing = [
                f
                for f in self.RAW_FEATURES
                if f not in chunk.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing features: {missing}"
                )

            # Numeric cleanup.
            for feature in self.RAW_FEATURES:

                chunk[feature] = pd.to_numeric(
                    chunk[feature],
                    errors="coerce",
                )

            chunk[self.RAW_FEATURES] = (
                chunk[self.RAW_FEATURES]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            for feature in self.RAW_FEATURES:

                median = chunk[feature].median()

                if pd.isna(median):
                    median = 0.0

                chunk[feature] = (
                    chunk[feature]
                    .fillna(median)
                )

            # Labels.
            canonical = (
                chunk["Label"]
                .map(self.canonical_label)
            )

            known = canonical.isin(
                self.label_to_stage
            )

            chunk = chunk.loc[known].copy()
            canonical = canonical.loc[known]

            chunk["mitre_stage"] = (
                canonical
                .map(self.label_to_stage)
                .astype(np.int64)
            )

            chunk["is_attack"] = (
                chunk["mitre_stage"] > 0
            ).astype(np.int64)

            chunks.append(
                chunk[
                    columns
                    + [
                        "mitre_stage",
                        "is_attack",
                    ]
                ]
            )

        if not chunks:

            raise RuntimeError(
                "No valid flows found in "
                + file_path
            )

        df = pd.concat(
            chunks,
            ignore_index=True,
        )

        df = df.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        print(
            f"    Flows: {len(df):,}"
        )

        print(
            f"    Time: {df.Timestamp.min()} "
            f"-> {df.Timestamp.max()}"
        )

        return df

    # =====================================================
    # FLOWS -> REAL 10-SECOND STATES
    # =====================================================

    def dataframe_to_states(
        self,
        df,
        window_seconds=10,
    ):

        df = df.copy()

        df["time_window"] = (
            df["Timestamp"]
            .dt.floor(
                f"{window_seconds}s"
            )
        )

        start = (
            df["time_window"].min()
        )

        end = (
            df["time_window"].max()
        )

        # Preserve empty time intervals.
        complete_timeline = pd.date_range(
            start=start,
            end=end,
            freq=f"{window_seconds}s",
        )

        grouped = {
            timestamp: group
            for timestamp, group
            in df.groupby("time_window")
        }

        states = []
        attacks = []
        stages = []

        for timestamp in complete_timeline:

            if timestamp not in grouped:

                # Empty 10-second state.
                state = np.zeros(
                    len(self.RAW_FEATURES) * 4 + 1,
                    dtype=np.float32,
                )

                attack = 0
                stage = 0

            else:

                window = grouped[timestamp]

                values = (
                    window[self.RAW_FEATURES]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                state = np.concatenate([
                    np.mean(values, axis=0),
                    np.std(values, axis=0),
                    np.max(values, axis=0),
                    np.min(values, axis=0),
                    np.array(
                        [len(window)],
                        dtype=np.float64,
                    ),
                ])

                state = np.nan_to_num(
                    state,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                # Fraction of malicious flows.
                malicious_ratio = float(
                    window["is_attack"].mean()
                )

                # State is malicious if >=10%
                # of flows are malicious.
                attack = int(
                    malicious_ratio >= 0.10
                )

                if attack:

                    malicious_stages = (
                        window.loc[
                            window["mitre_stage"] > 0,
                            "mitre_stage",
                        ]
                    )

                    stage = int(
                        malicious_stages
                        .mode()
                        .iloc[0]
                    )

                else:
                    stage = 0

            states.append(state)
            attacks.append(attack)
            stages.append(stage)

        states = np.asarray(
            states,
            dtype=np.float32,
        )

        attacks = np.asarray(
            attacks,
            dtype=np.int64,
        )

        stages = np.asarray(
            stages,
            dtype=np.int64,
        )

        print(
            f"    10-sec states: {len(states):,}"
        )

        print(
            f"    Attack states: "
            f"{100 * attacks.mean():.2f}%"
        )

        print(
            "    MITRE stages:",
            sorted(set(stages.tolist())),
        )

        return (
            states,
            attacks,
            stages,
        )

    # =====================================================
    # TEMPORAL SEQUENCES
    # =====================================================

    def states_to_sequences(
        self,
        states,
        labels,
        stages,
    ):

        sequence_length = int(
            self.config["data"][
                "sequence_length"
            ]
        )

        horizon = int(
            self.config["data"][
                "forecast_horizon"
            ]
        )

        X = []
        y_states = []
        y_attacks = []
        y_mitre = []

        limit = (
            len(states)
            - sequence_length
            - horizon
            + 1
        )

        for i in range(
            max(0, limit)
        ):

            history_end = (
                i + sequence_length
            )

            future_end = (
                history_end + horizon
            )

            X.append(
                states[i:history_end]
            )

            y_states.append(
                states[
                    history_end:
                    future_end
                ]
            )

            y_attacks.append(
                labels[
                    history_end:
                    future_end
                ]
            )

            y_mitre.append(
                stages[
                    history_end:
                    future_end
                ]
            )

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(
                y_states,
                dtype=np.float32,
            ),
            np.asarray(
                y_attacks,
                dtype=np.float32,
            ),
            np.asarray(
                y_mitre,
                dtype=np.int64,
            ),
        )

    # =====================================================
    # PROCESS A SEGMENT
    # =====================================================

    def process_segment(
        self,
        path,
        start_time=None,
        end_time=None,
    ):

        df = self.load_file(
            path,
            start_time,
            end_time,
        )

        return self.dataframe_to_states(
            df,
            window_seconds=int(
                self.config["data"]
                ["time_window_seconds"]
            ),
        )

    # =====================================================
    # TRAIN / VALIDATION / TEST
    # =====================================================

    def process_split(
        self,
        data_folder,
        train_files,
        validation_files,
        test_files,
        flows_per_window=None,
    ):
        """
        Timestamp-based split.

        Special handling for Wednesday-28:
        Early episode -> training
        Late episode  -> test

        There is a large temporal guard gap.
        """

        original_wed28 = (
            "Wednesday-28-02-2018_"
            "TrafficForML_CICFlowMeter.csv"
        )

        generated_train = (
            "Wednesday-28-INFIL-TRAIN.csv"
        )

        generated_test = (
            "Wednesday-28-INFIL-TEST.csv"
        )

        # -------------------------------------------------
        # Load ordinary files
        # -------------------------------------------------

        raw = {}

        def add_segment(
            key,
            path,
            start=None,
            end=None,
        ):

            states, labels, stages = (
                self.process_segment(
                    path,
                    start,
                    end,
                )
            )

            raw[key] = (
                states,
                labels,
                stages,
            )

        # =================================================
        # TRAIN
        # =================================================

        train_keys = []

        for filename in train_files:

            # Replace temporary row-split file
            # with original timestamp-split data.
            if filename == generated_train:

                key = "wed28_early_infiltration"

                add_segment(
                    key,
                    os.path.join(
                        data_folder,
                        original_wed28,
                    ),

                    # Early episode.
                    "2018-02-28 01:00:00",

                    # Stop well before later attack.
                    "2018-02-28 03:00:00",
                )

                train_keys.append(key)

            else:

                key = "train_" + filename

                add_segment(
                    key,
                    os.path.join(
                        data_folder,
                        filename,
                    ),
                )

                train_keys.append(key)

        # =================================================
        # VALIDATION
        # =================================================

        validation_keys = []

        for filename in validation_files:

            key = "validation_" + filename

            add_segment(
                key,
                os.path.join(
                    data_folder,
                    filename,
                ),
            )

            validation_keys.append(key)

        # =================================================
        # TEST
        # =================================================

        test_keys = []

        for filename in test_files:

            if filename == generated_test:

                key = "wed28_late_infiltration"

                add_segment(
                    key,
                    os.path.join(
                        data_folder,
                        original_wed28,
                    ),

                    # Several-hour temporal gap.
                    "2018-02-28 10:00:00",

                    "2018-02-28 13:00:00",
                )

                test_keys.append(key)

            else:

                key = "test_" + filename

                add_segment(
                    key,
                    os.path.join(
                        data_folder,
                        filename,
                    ),
                )

                test_keys.append(key)

        # =================================================
        # FIT SCALER ON TRAIN ONLY
        # =================================================

        training_states = np.concatenate(
            [
                raw[key][0]
                for key in train_keys
            ],
            axis=0,
        )

        print(
            "\n[*] Fitting scaler ONLY "
            "on training states..."
        )

        self.scaler.fit(
            training_states
        )

        self.scaler_fitted = True

        # =================================================
        # BUILD EACH SPLIT
        # =================================================

        def build(keys):

            X_parts = []
            ys_parts = []
            ya_parts = []
            ym_parts = []

            for key in keys:

                (
                    states,
                    labels,
                    stages,
                ) = raw[key]

                scaled = (
                    self.scaler
                    .transform(states)
                    .astype(np.float32)
                )

                (
                    X,
                    ys,
                    ya,
                    ym,
                ) = self.states_to_sequences(
                    scaled,
                    labels,
                    stages,
                )

                if len(X):

                    X_parts.append(X)
                    ys_parts.append(ys)
                    ya_parts.append(ya)
                    ym_parts.append(ym)

            return (
                np.concatenate(
                    X_parts,
                    axis=0,
                ),
                np.concatenate(
                    ys_parts,
                    axis=0,
                ),
                np.concatenate(
                    ya_parts,
                    axis=0,
                ),
                np.concatenate(
                    ym_parts,
                    axis=0,
                ),
            )

        train_data = build(
            train_keys
        )

        validation_data = build(
            validation_keys
        )

        test_data = build(
            test_keys
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "TIMESTAMP SPLIT SUMMARY"
        )

        print(
            "=" * 60
        )

        for name, dataset in [
            ("TRAIN", train_data),
            ("VALIDATION", validation_data),
            ("TEST", test_data),
        ]:

            X, _, ya, ym = dataset

            print(
                f"\n{name}"
            )

            print(
                "Sequences:",
                len(X),
            )

            print(
                "Shape:",
                X.shape,
            )

            print(
                "Attack ratio:",
                f"{100 * ya.mean():.2f}%",
            )

            print(
                "Stages:",
                sorted(
                    set(
                        ym.flatten().tolist()
                    )
                ),
            )

        return {
            "train":
                train_data,

            "validation":
                validation_data,

            "test":
                test_data,

            "feature_names":
                list(
                    self.RAW_FEATURES
                ),

            "scaler":
                self.scaler,

            "train_files":
                train_files,

            "validation_files":
                validation_files,

            "test_files":
                test_files,
        }