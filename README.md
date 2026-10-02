# Shepherd — Phase 1: Fraud Detection Platform Foundation

## Overview

**Shepherd** is an enterprise-oriented fraud detection platform designed to identify and manage potentially fraudulent financial transactions.

The project is being developed incrementally, with each phase establishing a production-grade component of the overall system.

**Phase 1 establishes the data and application foundation required for fraud detection.**

The goal of this phase is not yet to build the complete fraud decision engine. Instead, it establishes the core entities, persistence layer, database schema, and system structure that subsequent machine learning and real-time detection components will depend on.

---

## Phase 1 Objectives

Phase 1 focuses on building a reliable foundation for the fraud detection system.

### Primary objectives

* Establish the PostgreSQL database
* Define the core fraud-domain entities
* Create normalized relational tables
* Establish relationships between customers, accounts, devices, and merchants
* Create a persistent representation for fraud predictions
* Establish database migrations using Alembic
* Create a backend structure that can support subsequent ML and API layers
* Ensure the system can persist fraud-related data independently of the ML model

The key architectural principle is:

> **The machine learning model should be a component of the fraud detection system, not the system itself.**

---

# System Architecture

At the end of Phase 1, Shepherd is structured around a persistent transactional data layer.

```text
                    Shepherd
                       │
                       ▼
              ┌─────────────────┐
              │   Application   │
              │      Layer      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   PostgreSQL    │
              │    Database     │
              └────────┬────────┘
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   Customers        Accounts         Devices
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
                   Transactions
                       │
                       ▼
                  Fraud Results
                       │
                       ▼
             fraud_predictions
```

This separation allows the data layer to evolve independently from the machine learning layer.

---

# Database

Shepherd uses **PostgreSQL** as its primary relational database.

The database provides the persistent source of truth for the fraud detection platform.

The initial schema contains the following core tables:

```text
accounts
customers
devices
merchants
fraud_predictions
alembic_version
```

---

## Core Entities

### Customers

The `customers` table represents the people or business customers whose financial activity is being evaluated.

A customer may have multiple accounts and may interact with the system through multiple devices.

```text
Customer
   │
   ├── Account
   ├── Account
   └── Account
```

---

### Accounts

The `accounts` table represents financial accounts belonging to customers.

Accounts provide the connection between a customer and their transactional activity.

```text
Customer
    │
    └── Account
           │
           └── Transactions
```

This relationship is important because fraud patterns are often associated with account-level behavior rather than isolated transactions.

---

### Devices

The `devices` table represents devices associated with activity on the platform.

Device information provides an additional behavioral signal for fraud detection.

For example, future detection models may be able to identify patterns such as:

* Multiple accounts using the same device
* Unusual device-account relationships
* New devices appearing immediately before suspicious activity
* Device reuse across customers

The device entity therefore provides an important foundation for future behavioral and graph-based fraud features.

---

### Merchants

The `merchants` table represents merchants involved in financial transactions.

Merchant information allows Shepherd to eventually model patterns such as:

* Merchant-specific fraud rates
* Unusual customer-merchant relationships
* Transaction concentration
* Merchant risk profiles
* Geographic or behavioral anomalies

Merchant-level information becomes particularly valuable when combined with historical transaction data.

---

### Fraud Predictions

The `fraud_predictions` table represents the output of the fraud detection system.

This table is intentionally separated from the transactional entities.

The distinction is important:

```text
Transaction
     │
     ▼
Fraud Detection Model
     │
     ▼
Fraud Prediction
```

A prediction is an **assessment of a transaction**, rather than a property of the customer, account, merchant, or device itself.

This separation allows Shepherd to preserve prediction history and later support:

* Model scores
* Fraud classifications
* Model versions
* Prediction timestamps
* Investigation workflows
* Model monitoring
* Auditability

---

# Database Versioning

Shepherd uses **Alembic** for database schema migrations.

The `alembic_version` table tracks the current migration state.

This allows schema changes to be version-controlled rather than manually applied to the database.

The intended workflow is:

```text
Schema Change
     │
     ▼
Alembic Migration
     │
     ▼
PostgreSQL
```

This is important for reproducibility and makes it possible to evolve the database as Shepherd moves through subsequent phases.

---

# Why PostgreSQL?

PostgreSQL was selected as the initial persistence layer because Shepherd requires more than simple model storage.

The fraud detection system needs to represent relationships between:

```text
Customer
   │
   ├── Accounts
   │      │
   │      └── Transactions
   │
   ├── Devices
   │
   └── Merchants
```

A relational database provides a strong foundation for:

* Referential integrity
* Structured transactional data
* Relationships between entities
* Historical data
* Auditing
* SQL-based analytical queries
* Future feature engineering

The database is therefore treated as a core part of the fraud detection architecture rather than merely a storage mechanism for the ML model.

---

# Phase 1 Data Model

The conceptual relationship between the major entities is:

```text
                 ┌─────────────┐
                 │  Customer   │
                 └──────┬──────┘
                        │
                   owns │
                        ▼
                 ┌─────────────┐
                 │   Account   │
                 └──────┬──────┘
                        │
                 generates
                        │
                        ▼
                 ┌─────────────┐
                 │ Transaction │
                 └──────┬──────┘
                        │
                        ▼
                ┌────────────────┐
                │ Fraud Prediction│
                └────────────────┘

      ┌─────────────┐          ┌─────────────┐
      │   Device    │          │   Merchant  │
      └──────┬──────┘          └──────┬──────┘
             │                        │
             └──────── Transaction ───┘
```

The exact transactional relationships will expand as the system evolves.

---

# Design Principles

Phase 1 follows several principles that will remain important throughout Shepherd's development.

## 1. Data before intelligence

The ML model is only as reliable as the data infrastructure surrounding it.

Shepherd therefore establishes the data model before building increasingly sophisticated detection logic.

---

## 2. Separate predictions from source data

Fraud predictions should not overwrite the underlying transaction data.

Instead:

```text
Source Data
     │
     ├── Transaction
     ├── Customer
     ├── Account
     ├── Device
     └── Merchant
              │
              ▼
        Detection System
              │
              ▼
       Fraud Prediction
```

This makes predictions reproducible and allows different model versions to evaluate the same underlying transaction.

---

## 3. Preserve historical information

Fraud detection is inherently temporal.

A transaction cannot always be evaluated correctly in isolation.

Historical behavior can reveal:

* Changes in spending patterns
* New devices
* Unusual merchants
* Account behavior changes
* Repeated transaction patterns
* Relationships between entities

Phase 1 therefore establishes a persistent data layer capable of supporting historical analysis.

---

## 4. Design for future ML features

The Phase 1 schema is designed with future feature engineering in mind.

Later phases can derive features from relationships such as:

```text
Customer → Account
Account → Transaction
Transaction → Merchant
Transaction → Device
Customer → Device
```

These relationships can eventually support behavioral, statistical, temporal, and graph-based fraud signals.

---

# Migration State

The presence of the Alembic migration table confirms that database schema changes are being managed through migrations rather than unmanaged manual changes.

Current Phase 1 database foundation:

```text
PostgreSQL
    │
    ├── customers
    ├── accounts
    ├── devices
    ├── merchants
    ├── fraud_predictions
    └── alembic_version
```

---

# Phase 1 Deliverables

Phase 1 establishes the following foundation:

* [x] PostgreSQL database
* [x] Core customer entity
* [x] Account entity
* [x] Device entity
* [x] Merchant entity
* [x] Fraud prediction entity
* [x] Relational database structure
* [x] Alembic migration infrastructure
* [x] Persistent fraud-prediction storage

---

# What Phase 1 Does Not Attempt

Phase 1 intentionally does **not** attempt to solve the entire fraud detection problem.

The following components belong to subsequent development stages:

* Model training
* Feature engineering at scale
* Fraud probability calibration
* Real-time transaction scoring
* Decision thresholds
* Rule/model hybrid decisioning
* Model monitoring
* Drift detection
* Investigator workflows
* Alert management
* Model explainability
* Production deployment
* Real-time event processing

This separation keeps the development process incremental and makes each architectural layer independently testable.

---

# Next Phase

The next stage builds intelligence on top of the Phase 1 foundation.

The progression is:

```text
PHASE 1
Data & Persistence
        │
        ▼
PHASE 2
Feature Engineering & Fraud Model
        │
        ▼
PHASE 3
Fraud Scoring API
        │
        ▼
PHASE 4
Real-Time Detection Pipeline
        │
        ▼
PHASE 5
Monitoring, Explainability & Production Hardening
```

The objective is to evolve Shepherd from a database-backed application into a complete **production-grade fraud detection system**.

---

## Phase 1 Status

**Status:** Foundation established

**Primary infrastructure:** PostgreSQL + Alembic

**Core domain:** Customers, accounts, devices, merchants, and fraud predictions

**Focus:** Establishing the persistent data foundation for subsequent fraud detection and machine learning capabilities.


# Shepherd — Phase 2: Behavioral Simulation & Fraud Data Generation

## Overview

**Shepherd** is an enterprise-oriented fraud detection platform designed to detect suspicious financial activity using behavioral, transactional, and contextual signals.

Phase 1 established the persistence and domain foundation:

```text
Customers
Accounts
Devices
Merchants
Transactions
Fraud Predictions
Model Registry
```

**Phase 2 introduces realistic financial behavior into that foundation.**

The purpose of this phase is to generate a sufficiently realistic transaction environment in which fraud detection models can later be trained and evaluated.

Rather than beginning with an arbitrary synthetic dataset, Shepherd models customers as individuals with different behavioral profiles and generates transactions from those profiles. Fraud is then introduced into the transaction stream through explicit fraud-generation mechanisms.

---

# Phase 2 Objectives

The primary objectives of Phase 2 are:

- Generate a realistic customer population
- Assign customers behavioral archetypes
- Generate customer accounts and devices
- Generate merchants
- Simulate normal transaction behavior
- Introduce fraudulent transaction behavior
- Persist transactions into PostgreSQL
- Maintain referential integrity across entities
- Produce a labeled dataset suitable for machine learning
- Validate the generated data before feature engineering

The overall pipeline is:

```text
Customer Population
        │
        ▼
Customer Behavioral Profiles
        │
        ▼
Accounts + Devices
        │
        ▼
Merchant Population
        │
        ▼
Normal Transaction Generation
        │
        ▼
Fraud Injection
        │
        ▼
Validated Transaction Dataset
        │
        ▼
PostgreSQL
```

---

# Why Simulation?

Real-world fraud datasets are difficult to obtain because transaction data is highly sensitive.

Public datasets also frequently lack the combination of:

- Customer identity
- Account relationships
- Device information
- Merchant information
- Transaction history
- Behavioral context
- Ground-truth fraud labels

Shepherd therefore uses a **behavioral simulation approach**.

The objective is not to reproduce a particular financial institution's transaction data.

Instead, the simulator creates controlled relationships and behavioral patterns that allow the fraud detection pipeline to be developed and tested.

This gives Shepherd control over:

- Fraud prevalence
- Customer behavior
- Transaction volume
- Device usage
- Geographic behavior
- Spending patterns
- Temporal behavior
- Fraud scenarios

---

# Customer Behavioral Model

Customers are not generated as identical random users.

Each customer is assigned a behavioral archetype that controls their expected transaction behavior.

The current archetypes are:

| Archetype | Transactions / Day | Average Spend | International Probability | Devices |
|---|---:|---:|---:|---:|
| Conservative | 1–4 | $35 | 2% | 1–2 |
| Professional | 3–10 | $75 | 10% | 2–4 |
| Traveler | 4–12 | $120 | 45% | 2–4 |
| High Net Worth | 2–6 | $600 | 30% | 2–5 |
| Digital Native | 5–20 | $40 | 15% | 3–6 |

These archetypes allow Shepherd to generate heterogeneous behavior rather than treating every customer as statistically identical.

---

# Customer State

Each simulated customer maintains behavioral state.

Conceptually:

```text
CustomerState
│
├── customer_id
├── archetype
├── home_country
├── average_spend
├── spend_std
├── transactions_per_day
├── usual_countries
├── known_devices
├── preferred_merchant_categories
└── last_transaction_timestamp
```

This state is important because fraud detection depends on deviations from an individual's historical behavior.

For example:

```text
Customer normally:
$20–$60 transactions
        │
        ▼
Sudden $1,200 transaction
        │
        ▼
Behavioral deviation
```

The simulator therefore provides the historical context required for later feature engineering.

---

# Population Generation

The population generator creates customers according to the defined behavioral archetypes.

For example, a population generated with a fixed random seed can be reproduced deterministically.

A 10,000-customer population produced with seed `42` had the following distribution:

| Archetype | Customers | Proportion |
|---|---:|---:|
| Conservative | 3,028 | 30.28% |
| Professional | 3,026 | 30.26% |
| Digital Native | 1,524 | 15.24% |
| Traveler | 1,449 | 14.49% |
| High Net Worth | 973 | 9.73% |
| **Total** | **10,000** | **100%** |

Deterministic seeds make the simulation reproducible during development and testing.

---

# Transaction Simulation

Once customers, accounts, devices, and merchants exist, Shepherd generates transactions based on the customer's behavioral state.

A normal transaction is influenced by factors such as:

```text
Customer
   │
   ├── Spending behavior
   ├── Transaction frequency
   ├── Geographic behavior
   ├── Known devices
   └── Merchant preferences
            │
            ▼
       Transaction
```

Each transaction contains the information required to reconstruct its context.

The transaction model includes:

- Transaction ID
- Account ID
- Merchant ID
- Device ID
- Amount
- Currency
- Timestamp
- IP address
- Fraud label
- Creation timestamp

---

# Fraud Injection

Normal behavior alone does not produce a useful supervised fraud-detection dataset.

Phase 2 therefore introduces fraudulent activity into the simulated transaction stream.

The fraud-generation layer is designed around **behavioral anomalies**, rather than simply assigning random transactions as fraudulent.

This distinction is important.

A useful fraud dataset should allow a model to learn patterns such as:

```text
Normal Customer Behavior
          │
          ▼
   Behavioral Baseline
          │
          ▼
    Suspicious Event
          │
          ▼
      Fraud Label
```

The resulting labels can later be used to train supervised fraud models.

---

# Transaction Dataset

The current Phase 2 simulation has successfully generated and persisted:

| Metric | Result |
|---|---:|
| Total transactions | **2,932,534** |
| Fraudulent transactions | **146,646** |
| Non-fraudulent transactions | **2,785,888** |
| Unlabeled transactions | **0** |
| Fraud rate | **5.00%** |
| Time range | **2026-06-02 → 2026-08-29** |

The dataset therefore contains nearly **3 million labeled transactions**.

The 5% fraud prevalence is intentionally controlled by the simulation rather than representing a claim about real-world financial fraud prevalence.

---

# Data Integrity Validation

Before moving to machine learning, the generated transaction dataset is validated against the database constraints and expected simulation properties.

The current validation results are:

| Validation | Result |
|---|---:|
| Duplicate transactions | **0** |
| Orphaned references | **0** |
| NULL/invalid countries | **0** |
| Unlabeled transactions | **0** |
| Fraud transactions | **146,646** |
| Non-fraud transactions | **2,785,888** |

The absence of orphaned records is particularly important because transactions depend on previously persisted entities:

```text
Customer
   │
   ▼
Account
   │
   ├──────────────┐
   ▼              ▼
Transaction    Customer
   │
   ├── Merchant
   │
   └── Device
```

This ensures that downstream feature engineering can rely on valid relational history.

---

# Persistence Architecture

Transactions are persisted directly into the PostgreSQL database established during Phase 1.

The resulting architecture is:

```text
             Simulation
                 │
                 ▼
       ┌──────────────────┐
       │ Transaction      │
       │ Generator        │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Validation       │
       │ & Integrity      │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ PostgreSQL       │
       │                  │
       │ transactions     │
       └──────────────────┘
```

The database therefore becomes the persistent source of truth for the simulated financial activity.

---

# Why Behavioral Simulation Matters

The most important property of the simulator is that transactions are generated **within a behavioral context**.

This allows future features to be derived from historical behavior.

For example:

### Velocity

```text
Number of transactions
within the last 10 minutes
```

### Spending deviation

```text
Current amount
        │
        ▼
Customer historical average
```

### Geographic deviation

```text
Current country
        │
        ▼
Home / usual countries
```

### Device novelty

```text
Current device
        │
        ▼
Previously known devices
```

### Merchant behavior

```text
Current merchant/category
        │
        ▼
Customer's historical preferences
```

These signals cannot be meaningfully generated from isolated random transactions.

---

# Phase 2 → Machine Learning Pipeline

The next component built on this dataset is the feature engineering layer.

The intended flow is:

```text
PostgreSQL
    │
    ▼
Historical Transactions
    │
    ▼
Feature Engineering
    │
    ├── Transaction features
    ├── Velocity features
    ├── Behavioral deviation
    ├── Device features
    ├── Geographic features
    ├── Merchant features
    └── Temporal features
    │
    ▼
Training Dataset
    │
    ▼
Fraud Detection Model
    │
    ▼
Model Evaluation
```

---

# Planned Feature Groups

The next stage will derive leakage-safe features from the persisted transaction history.

### Transaction features

Examples:

- Transaction amount
- Amount relative to customer historical average
- Amount deviation from expected spending

### Velocity features

Examples:

- Transaction count in the last 10 minutes
- Transaction count in the last hour
- Transaction count in the last 24 hours

### Device features

Examples:

- Device age
- Whether the device is new
- Historical device usage

### Geographic features

Examples:

- Home-country indicator
- Usual-country indicator
- International transaction indicator

### Merchant features

Examples:

- Merchant risk score
- Merchant category
- Customer familiarity with merchant/category

### Temporal features

Examples:

- Time since previous transaction
- Transaction hour
- Historical transaction patterns

The critical requirement is that features must be calculated using **information available before the transaction being scored**.

---

# Leakage Prevention

Fraud detection is particularly vulnerable to temporal leakage.

A feature must not accidentally use information from the future.

For example, when evaluating transaction `T`:

```text
T-3 ── T-2 ── T-1 ── T
                  │
                  └── Available information
```

The model may use information from `T-1` and earlier.

It must not use:

```text
T+1
T+2
T+3
```

to construct features for `T`.

This principle will be enforced during the feature-engineering stage.

---

# Model Development

Once the feature pipeline is complete, Shepherd will train supervised fraud detection models.

The planned modeling process is:

```text
Feature Dataset
      │
      ▼
Time-Aware Split
      │
      ├───────────────┐
      ▼               ▼
Baseline Model    Tree Model
      │               │
      ▼               ▼
Evaluation      Evaluation
      │               │
      └───────┬───────┘
              ▼
       Model Comparison
```

The initial model development strategy includes:

- Logistic Regression baseline
- XGBoost
- Time-aware validation
- Precision-recall analysis
- ROC-AUC
- PR-AUC
- Recall at a target precision
- Business-cost analysis

Model selection will be based on the operational objective of fraud detection rather than a single generic metric.

---

# Model Registry

Phase 1 also established the database structures required to track model outputs and model versions.

The `fraud_predictions` table is designed to preserve prediction-level information, including:

```text
transaction_id
model_version
fraud_probability
decision
threshold
timestamp
```

The `model_registry` table records model-level metadata such as:

```text
model_version
training_date
AUC-PR
recall_at_target_precision
threshold
artifact_path
approval
```

This provides the foundation for model versioning and auditability.

---

# Current Phase 2 Status

### Completed

- [x] Customer behavioral archetypes
- [x] Customer population generation
- [x] Account generation
- [x] Device generation
- [x] Merchant generation
- [x] Normal transaction simulation
- [x] Fraud injection
- [x] Transaction persistence
- [x] Fraud labeling
- [x] Database integrity validation
- [x] Nearly 3 million persisted transactions

### Next

- [ ] Leakage-safe feature engineering
- [ ] Historical behavioral feature generation
- [ ] Training dataset construction
- [ ] Time-aware model split
- [ ] Logistic Regression baseline
- [ ] XGBoost model
- [ ] Fraud model evaluation
- [ ] Threshold selection
- [ ] Model artifact/version registration

---

# Phase 2 Definition of Done

Phase 2 establishes a sufficiently realistic and validated transaction environment for machine learning.

The phase is considered complete when Shepherd has:

```text
✓ Realistic customer behavior
✓ Persistent financial activity
✓ Fraudulent behavior
✓ Ground-truth labels
✓ Referential integrity
✓ Historical transaction context
✓ Reproducible simulation
✓ ML-ready transaction data
```

The current transaction-generation milestone has been achieved:

> **2,932,534 transactions have been generated, persisted, and validated, with 146,646 labeled as fraudulent and zero unlabeled transactions.**

---

# Phase 3 Preview

The next stage moves from **data generation to intelligence**.

```text
PHASE 1
Data & Persistence
        │
        ▼
PHASE 2
Behavioral Simulation
        │
        ▼
PHASE 3
Feature Engineering & Fraud Model
        │
        ▼
PHASE 4
Real-Time Fraud Scoring
        │
        ▼
PHASE 5
Monitoring & Production Hardening
```

The objective is to transform the simulated transaction environment into a machine-learning-powered fraud detection system capable of assigning meaningful risk scores to individual transactions.

---

## Engineering Philosophy

Shepherd is being developed around several principles:

### 1. Build the system, not just the model

The ML model is one component of a larger fraud detection platform.

### 2. Simulate behavior, not random numbers

Synthetic data should contain relationships that make behavioral fraud detection possible.

### 3. Preserve temporal causality

Features must represent what was knowable at the time of the transaction.

### 4. Validate before modeling

A model trained on structurally invalid data produces misleading results.

### 5. Measure operationally

Fraud detection involves asymmetric costs. False negatives and false positives do not have equivalent consequences.

### 6. Make the system auditable

Transactions, predictions, model versions, thresholds, and evaluation results should be traceable.

---

**Status:** Phase 2 transaction simulation completed.  
**Dataset:** 2,932,534 persisted transactions.  
**Fraud labels:** 146,646 fraud / 2,785,888 non-fraud / 0 unlabeled.  
**Next milestone:** Leakage-safe feature engineering and fraud-model development.

# Shepherd — Phase 3: Fraud Feature Engineering & ML Dataset

## Overview

**Shepherd** is an enterprise-oriented fraud detection platform designed to identify suspicious financial transactions using behavioral, transactional, device, merchant, and temporal signals.

Phase 1 established the **banking data model and persistence layer**.

Phase 2 populated that environment with a **synthetic financial population and nearly three million labeled transactions**.

Phase 3 transforms that raw transaction history into a **machine-learning-ready fraud feature dataset**.

The central problem in this phase was not simply:

> "How do we create features?"

It was:

> **"How do we create features using only information that would have been available at the moment a transaction occurred?"**

That requirement makes temporal correctness and leakage prevention the primary design constraints of Phase 3.

---

# Phase 3 Objectives

Phase 3 was designed to:

* Convert raw transaction history into ML features
* Preserve chronological causality
* Prevent future information from leaking into historical features
* Generate customer behavioral features
* Generate transaction velocity features
* Generate device and merchant context
* Generate historical spending features
* Produce a reproducible feature dataset
* Process the full transaction population without requiring the entire dataset in memory
* Validate the resulting dataset before model training

The completed pipeline is:

```text
                    PostgreSQL
                        │
                        ▼
                 Raw Transactions
                        │
                        ▼
             Chronological Processing
                        │
                        ▼
                 FeatureEngine
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        Customer     Transaction   Context
        History       Velocity     Signals
            │           │           │
            └───────────┼───────────┘
                        │
                        ▼
              Feature Dataset
                        │
                        ▼
             Automated Validation
                        │
                        ▼
        transaction_features.parquet
```

---

# The Core Design Problem: Temporal Leakage

Fraud detection is fundamentally different from many standard tabular ML problems.

At transaction time `T`, Shepherd can only use information that was available **before or at `T`**.

For example:

```text
T-3 ───── T-2 ───── T-1 ───── T
                                  ▲
                            Transaction
                            being scored
```

Valid information:

```text
T-3
T-2
T-1
```

Invalid information:

```text
T+1
T+2
T+3
```

A feature such as:

```text
customer_average_transaction_amount
```

must therefore represent the customer's historical average **before the current transaction**.

It cannot include the current transaction itself.

Otherwise the model would receive information about the event it is supposed to predict.

---

# Chronological Feature Generation

Shepherd processes transactions globally in chronological order.

The feature pipeline streams transactions from PostgreSQL ordered by:

```text
transaction_timestamp
transaction_id
```

This establishes deterministic chronological processing.

Conceptually:

```text
Transaction 1
     │
     ▼
Update historical state
     │
     ▼
Transaction 2
     │
     ▼
Update historical state
     │
     ▼
Transaction 3
     │
     ▼
...
```

The important rule is:

> **Generate features first → then update historical state.**

This ensures that the current transaction cannot influence its own features.

---

# Stateful Feature Engineering

The `FeatureEngine` maintains historical state while processing the transaction stream.

For each customer, Shepherd can maintain information such as:

```text
Customer History
│
├── Previous transaction timestamp
├── Number of previous transactions
├── Historical amount statistics
├── Known merchants/categories
└── Recent transaction timestamps
```

This state is updated as transactions are processed.

The result is a feature vector representing the customer's state **immediately before the transaction**.

---

# Feature Dataset

The completed feature dataset contains:

| Metric                      |         Value |
| --------------------------- | ------------: |
| Rows                        | **2,932,534** |
| Unique transactions         | **2,932,534** |
| Fraudulent transactions     |   **146,646** |
| Non-fraudulent transactions | **2,785,888** |
| Fraud rate                  |   **5.0007%** |
| Target                      |    `is_fraud` |

Every source transaction therefore maps to exactly one feature row.

```text
1 transaction
      │
      ▼
1 feature vector
      │
      ▼
1 fraud label
```

---

# Feature Groups

The Phase 3 dataset contains features covering several dimensions of transaction behavior.

## 1. Transaction Amount

### `amount`

The raw transaction amount.

This provides the model with the absolute monetary value of the transaction.

Example:

```text
Customer normally:
$20 → $80

Current transaction:
$1,200
```

The raw amount alone is not enough to determine whether this is suspicious.

That is why Shepherd also generates behavioral features.

---

# 2. Merchant Risk

### `merchant_risk_score`

A merchant-level risk signal associated with the transaction.

This allows the model to distinguish between transactions occurring in different merchant risk environments.

Conceptually:

```text
Transaction
     │
     ▼
Merchant
     │
     ▼
Merchant Risk Signal
```

Merchant context can become especially valuable when combined with customer history.

---

# 3. Geographic Behavior

### `is_home_country`

Indicates whether the transaction occurred in the customer's home country.

This creates a basic geographic behavioral signal:

```text
Customer home country
        │
        ▼
Current transaction country
        │
        ├── Match → expected
        │
        └── Different → potentially unusual
```

This feature is deliberately simple.

The model can later combine it with other signals rather than treating international activity as inherently fraudulent.

---

# 4. Device Behavior

### `device_age_hours`

Represents the age of the device associated with the transaction.

Device age provides temporal context around device usage.

For example:

```text
Known device
     vs.
Very recently introduced device
```

A new device may be perfectly legitimate, but it can become informative when combined with other behavioral signals.

---

# 5. Transaction History

### `transactions_before`

Number of transactions previously observed for the customer.

This provides a basic measure of historical activity.

```text
transactions_before = 0
```

indicates a customer's first observed transaction.

While:

```text
transactions_before = 450
```

indicates a customer with substantial historical behavior.

This distinction becomes important when interpreting other features.

---

# 6. Historical Spending

### `avg_amount_before`

The customer's average transaction amount **before the current transaction**.

This feature establishes a behavioral baseline.

Example:

```text
Historical average = $45
Current transaction = $500
```

The absolute amount is useful.

The deviation from the customer's own behavior is often more informative.

---

# 7. Amount Deviation

### `amount_vs_avg`

Measures the current transaction amount relative to the customer's historical average.

Conceptually:

```text
Current Amount
      │
      ▼
Historical Average
      │
      ▼
Behavioral Deviation
```

This allows Shepherd to distinguish between:

```text
$500 transaction for a customer who normally spends $600
```

and:

```text
$500 transaction for a customer who normally spends $30
```

The same absolute amount can therefore represent very different behavioral contexts.

---

# 8. Time Since Previous Transaction

### `seconds_since_previous_tx`

Measures the time elapsed since the customer's previous transaction.

This enables Shepherd to identify transaction timing patterns.

For example:

```text
Transaction A
     │
     │  2 days
     ▼
Transaction B
```

versus:

```text
Transaction A
     │
     │  8 seconds
     ▼
Transaction B
```

The latter may become important when combined with transaction velocity and other signals.

### First-transaction behavior

There are **10,000 expected NULL values** for this feature.

These correspond to customers for whom Shepherd has no previous transaction.

This is not treated as an unexpected data-quality failure.

It is a legitimate representation of:

```text
First observed transaction
        ↓
No previous transaction
        ↓
seconds_since_previous_tx = NULL
```

---

# 9. Short-Term Transaction Velocity

Shepherd generates transaction counts across multiple time windows.

### `tx_count_10m`

Number of transactions in the previous 10 minutes.

### `tx_count_1h`

Number of transactions in the previous hour.

### `tx_count_24h`

Number of transactions in the previous 24 hours.

This gives the model multiple temporal resolutions.

```text
                Customer activity
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      10 min          1 hour         24 hours
```

This matters because suspicious activity can appear as either:

* A sudden burst over seconds/minutes
* Unusual hourly activity
* An abnormal increase in daily activity

---

# 10. Merchant Familiarity

### `merchant_category_seen_before`

Indicates whether the customer has previously interacted with the relevant merchant category.

This creates a behavioral familiarity signal.

Conceptually:

```text
Customer history
      │
      ▼
Merchant categories previously seen
      │
      ▼
Current merchant category
      │
      ▼
Seen before?
```

A previously unseen category is not automatically fraudulent.

It is simply an additional signal that the model can combine with other evidence.

---

# Final Feature Set

The completed Phase 3 dataset contains the following core feature columns:

```text
amount
merchant_risk_score
is_home_country
device_age_hours
transactions_before
avg_amount_before
amount_vs_avg
seconds_since_previous_tx
tx_count_10m
tx_count_1h
tx_count_24h
merchant_category_seen_before
```

Target:

```text
is_fraud
```

---

# Leakage-Safe Feature Construction

The most important architectural property of Phase 3 is the ordering of operations.

For each transaction:

```text
1. Read transaction
       │
       ▼
2. Read historical state
       │
       ▼
3. Generate features
       │
       ▼
4. Write feature row
       │
       ▼
5. Update historical state
       │
       ▼
6. Move to next transaction
```

The order is intentional.

If Shepherd instead performed:

```text
1. Update state
2. Generate features
```

the current transaction could contaminate its own historical statistics.

That would create temporal leakage.

---

# Memory-Efficient Processing

The dataset contains almost three million transactions.

Loading the entire transaction table and all intermediate feature state into memory would unnecessarily increase resource requirements.

Phase 3 therefore uses a streaming approach.

```text
PostgreSQL
     │
     ▼
Transaction stream
     │
     ▼
FeatureEngine
     │
     ▼
Bounded historical state
     │
     ▼
Parquet output
```

The generated dataset is written as a compressed **Zstandard (Zstd) Parquet** artifact.

Output:

```text
artifacts/features/transaction_features.parquet
```

This provides an efficient columnar representation suitable for downstream ML workloads.

---

# Why Parquet?

Parquet is well suited to the feature dataset because:

* It is columnar
* Numeric feature columns can be read selectively
* It provides efficient compression
* It is substantially more appropriate for ML datasets than repeatedly parsing CSV
* It works well with Python data tooling
* It provides a clean boundary between feature generation and model training

The feature pipeline therefore produces a durable ML artifact rather than requiring feature computation to be repeated every time a model is trained.

---

# Feature Dataset Audit

After generation, Shepherd runs an automated feature-dataset audit.

The audit checks several classes of failure.

## Row integrity

```text
Rows:
2,932,534

Unique transaction IDs:
2,932,534
```

Therefore:

```text
1 row ↔ 1 transaction
```

No duplicate transaction feature rows were detected.

---

## Target integrity

The dataset contains:

```text
Fraud:
146,646

Non-fraud:
2,785,888
```

Fraud rate:

```text
5.0007%
```

This matches the underlying simulated transaction population.

---

# Null Validation

The audit reports:

```text
Unexpected null values:
0
```

The only expected missing values are the:

```text
10,000
```

`seconds_since_previous_tx` values associated with first transactions.

This distinction is important.

The validation system does not simply demand:

> "There must be zero NULLs."

Instead, it asks:

> **"Are the NULLs expected and explainable?"**

That is a more appropriate approach for production feature pipelines.

---

# Numeric Validation

The audit also checks numeric features for non-finite values.

Result:

```text
Non-finite numeric values:
0
```

No unexpected:

```text
NaN
+∞
-∞
```

values were detected.

---

# Feature Range Validation

Expected feature ranges were checked against the generated dataset.

Result:

```text
Feature range checks:
PASS
```

This protects the downstream model-training pipeline from obvious corrupted or impossible feature values.

---

# Chronological Validation

The feature dataset was also checked for chronological consistency.

The feature-generation process is explicitly chronological, and the audit verifies that the resulting data preserves the intended transaction ordering.

This is critical because the validity of historical features depends on temporal ordering.

---

# Phase 3 Architecture

The completed architecture can be summarized as:

```text
                    PostgreSQL
                        │
                        │ chronological stream
                        ▼
              ┌──────────────────┐
              │   FeatureEngine  │
              └────────┬─────────┘
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
   Customer         Velocity        Context
    History         Features        Features
        │              │               │
        └──────────────┼───────────────┘
                       │
                       ▼
             Feature Dataset
                       │
                       ▼
          Automated Feature Audit
                       │
                       ▼
     transaction_features.parquet
```

---

# Phase 1 → Phase 2 → Phase 3

Shepherd is now developing as a layered system.

## Phase 1 — Banking Infrastructure

Built the synthetic bank's persistent structure.

```text
Customers
Accounts
Devices
Merchants
Transactions
Fraud Predictions
Model Registry
```

---

## Phase 2 — Synthetic Financial Activity

Populated the bank with realistic behavioral activity.

```text
10,000 customers
       │
       ▼
Accounts + Devices + Merchants
       │
       ▼
2,932,534 transactions
       │
       ▼
146,646 fraud labels
```

---

## Phase 3 — Machine Learning Representation

Converted raw transactions into model-readable behavioral representations.

```text
2,932,534 transactions
          │
          ▼
Temporal Feature Engine
          │
          ▼
Behavioral Features
          │
          ▼
Validated Feature Dataset
          │
          ▼
transaction_features.parquet
```

The result is the first point at which Shepherd has a proper **ML-ready representation of its synthetic bank**.

---

# Why Phase 3 Matters

The database tells Shepherd:

> **What happened?**

Phase 3 begins to answer:

> **What was normal for this customer before it happened?**

For example:

```text
RAW TRANSACTION

Amount: $850
```

is relatively weak information.

Phase 3 can transform that into:

```text
Amount:                 $850
Customer average:       $42
Amount vs average:      High
Transactions in 10 min: 5
Transactions in 1 hour: 8
Device age:             2 hours
Merchant category:     Previously unseen
Home country:           No
```

The transaction is no longer just a row in a database.

It has become a **behavioral event**.

That is the representation a fraud model can reason about.

---

# Phase 3 Validation Summary

The completed feature-generation pipeline produced:

```text
Rows                         2,932,534
Unique transaction IDs      2,932,534
Fraud                       146,646
Non-fraud                   2,785,888
Fraud rate                  5.0007%

Unexpected NULLs            0
Expected first-tx NULLs     10,000
Non-finite numeric values   0
Feature range checks        PASS
Chronological validation    PASS
```

This gives Shepherd a validated foundation for model development.

---

# Output Artifact

The primary Phase 3 artifact is:

```text
artifacts/
└── features/
    └── transaction_features.parquet
```

This file represents the boundary between:

```text
DATA ENGINEERING
        │
        ▼
FEATURE ENGINEERING
        │
        ▼
──────── PHASE 3 OUTPUT ────────
        │
        ▼
MACHINE LEARNING
```

---

# What Phase 3 Does Not Do

Phase 3 deliberately stops at the creation and validation of the ML feature dataset.

It does **not** yet represent the final fraud model.

The following belong to the next stage:

* Chronological train/validation/test splitting
* Model training
* Baseline model development
* XGBoost training
* Model comparison
* PR-AUC evaluation
* ROC-AUC evaluation
* Recall at target precision
* Threshold selection
* Model artifact registration
* Production fraud scoring

This separation is intentional.

A reliable fraud model should be built **after** the feature pipeline has been independently validated.

---

# Phase 4 Preview

The next stage consumes the Phase 3 artifact:

```text
transaction_features.parquet
             │
             ▼
      Chronological Split
             │
       ┌─────┴─────┐
       ▼           ▼
   Training     Validation
       │           │
       └─────┬─────┘
             ▼
       Fraud Models
             │
             ▼
       Model Evaluation
             │
             ▼
       Selected Model
             │
             ▼
       Model Registry
```

The objective is to teach Shepherd how to distinguish normal transactions from fraudulent ones using the behavioral representation constructed in Phase 3.

---

# Engineering Principles

Phase 3 reinforces several principles that will remain central to Shepherd.

## 1. Temporal causality over convenience

A feature is only valid if it could have existed when the transaction occurred.

## 2. Behavior over isolated values

A transaction amount becomes more meaningful when compared with the customer's history.

## 3. Explicit state

Historical customer behavior is maintained explicitly rather than reconstructed inconsistently during training.

## 4. Streaming over brute force

Large transaction populations should not require loading the entire dataset into memory.

## 5. Validate the data before validating the model

A model cannot compensate for a broken feature pipeline.

## 6. Separate feature generation from model training

The feature dataset is a reusable artifact that can support multiple experiments and models.

---

# Phase 3 Status

**Status: COMPLETE**

### Delivered

* [x] Chronological transaction streaming
* [x] Stateful feature engineering
* [x] Leakage-safe historical features
* [x] Transaction amount features
* [x] Merchant risk features
* [x] Geographic features
* [x] Device features
* [x] Customer historical behavior
* [x] Transaction velocity features
* [x] Merchant familiarity features
* [x] Full 2,932,534-row feature dataset
* [x] Zstd-compressed Parquet artifact
* [x] Automated feature audit
* [x] Duplicate-ID validation
* [x] Null validation
* [x] Numeric finite-value validation
* [x] Feature range validation
* [x] Chronological validation

---

# Final Phase 3 Outcome

Phase 1 gave Shepherd a **bank**.

Phase 2 gave the bank **customers and financial activity**.

Phase 3 gave Shepherd the ability to describe that activity in terms of **behavior**.

```text
                    SHEPHERD

       ┌─────────────────────────────┐
       │       SYNTHETIC BANK        │
       │                             │
       │ Customers                   │
       │ Accounts                    │
       │ Devices                     │
       │ Merchants                   │
       │ Transactions                │
       └──────────────┬──────────────┘
                      │
                      ▼
             HISTORICAL BEHAVIOR
                      │
                      ▼
             FEATURE ENGINEERING
                      │
                      ▼
           ┌──────────────────────┐
           │ ML FEATURE DATASET    │
           │                      │
           │ 2,932,534 rows       │
           │ 12 core features     │
           │ 146,646 fraud labels │
           └──────────┬───────────┘
                      │
                      ▼
                PHASE 4
              FRAUD MODEL
```

The key achievement of Phase 3 is therefore not simply the creation of twelve columns.

It is the establishment of a **leakage-safe, reproducible feature-generation system that converts Shepherd's simulated banking history into a representation suitable for machine learning.**

**Phase 3 complete.**
