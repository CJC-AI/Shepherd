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
