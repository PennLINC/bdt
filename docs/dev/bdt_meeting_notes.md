# Ad Hoc BDT Meeting 03/23/26

Type: Software Development
Date: March 23, 2026
Goal: Discuss scope and structure of BDT
Created: April 13, 2026 11:31 AM
Last Edited Time: April 13, 2026 11:39 AM
Participants: Taylor Salo, Matt Cieslak
Topics: BDT

# Goals

- 

# Discussion Items

- 

# Action Items

- [ ]  

# Transcript Summary

Here is a structured, decision-focused summary of the meeting on defining the scope and structure of the **BDT (BIDS Derivative Transformer)**.

---

## 1. High-level Goal of BDT

BDT is intended as a **general-purpose transformation layer over BIDS derivatives**, with the core function:

- **Inputs**:
    - BIDS derivative datasets (e.g., fMRI, dMRI outputs)
    - BIDS atlas datasets
- **Operations**:
    - Apply atlases to derivatives (parcellation, aggregation)
- **Outputs**:
    - Tabular data (e.g., time series, scalar summaries)
    - Potentially connectivity matrices

It is explicitly designed to:

- Support **volumetric, surface, and hybrid data**
- Work across **native and standard spaces**
- Handle **scalar maps, time series, and streamlines**

---

## 2. Key Decisions

### 2.1 Separation of Responsibilities (Major Architectural Decision)

**Decision: Split functionality into two tools**

- **BDT** → applies atlases to data
- **BAT (BIDS Atlas Transformer)** → manipulates/creates atlases

Rationale:

- Atlas manipulation (e.g., intersections, unions) is fundamentally different from applying atlases
- Keeps BDT simpler and focused

---

### 2.2 Initial Scope Reduction

**Decision: Defer complex features**

- Drop or postpone:
    - Complex atlas composition (“flat composition”)
    - Advanced cross-modal atlas combinations
- Focus on:
    - “Readily available” transformations first

---

### 2.3 Core Output Types

**Decision: Prioritize tabular outputs**

- Parcel-wise scalar summaries (e.g., mean values)
- Time series (e.g., parcel or bundle time series)
- Formats:
    - TSV / Parquet (for scalability)

---

### 2.4 Transform Strategy

**Decision: Transform atlases → NOT data**

- Find transform chain
- Apply transforms to atlas
- Then extract data

This avoids:

- Repeated interpolation of data
- Complexity of transforming multiple modalities

---

### 2.5 CLI Design Direction

**Decision: Use a dataset-centric CLI**

- `-datasets` includes:
    - Derivatives
    - Atlas datasets
- Users specify:
    - Atlases
    - Derivatives to process

---

### 2.6 User Explicitness Over Heuristics

**Decision: Prefer explicit user specification**

- If multiple valid inputs exist → raise error
- Avoid implicit “best choice” heuristics

---

### 2.7 Config / Grammar Approach

**Decision: Use structured config (not pure CLI flags)**

- Inspired by:
    - QSIRecon “recon spec”
    - FitLins-style transformations
- Enables:
    - Multi-step transformations
    - Reusable “actions” (e.g., union, intersection)

---

### 2.8 Transform Resolution Strategy

**Decision: Use a graph-based transform system**

- Represent transforms as a graph
- Use shortest-path search to find transform chains

---

### 2.9 Priority System for Matching Data

**Decision: Define priority ordering**

- Space
- Resolution
- Format

Used to select appropriate inputs/targets

---

## 3. Answered Questions

### Q: What is the minimal viable interface?

**Answer:**

- Input datasets + atlas datasets
- Lists of derivatives + atlases
- Output: tabular summaries

---

### Q: Should atlas transformations be part of BDT?

**Answer:**

- No → separate tool (BAT)

---

### Q: Should BDT choose between multiple valid inputs automatically?

**Answer:**

- No → user must disambiguate

---

### Q: Where should transforms be applied?

**Answer:**

- To atlases, not data

---

### Q: What data types are supported?

**Answer:**

- Volumes, surfaces, streamlines, hybrid data

---

## 4. Open Questions

### 4.1 Transform Selection Heuristics

- Should there ever be automatic selection vs strict errors?
- How sophisticated should priority rules be?

---

### 4.2 Config Specification Design

- Exact schema for:
    - Entities vs named objects
    - Transform steps
- How close to:
    - BIDS filter files?
    - QSIRecon specs?

---

### 4.3 Atlas Representation

- How to handle:
    - Subject-specific atlases (e.g., bundles)
    - Non-precomputable atlases

---

### 4.4 Output Space Decisions

- Where to specify:
    - Output space vs input space
- How to handle multiple valid mappings

---

### 4.5 Performance / Scalability

- Large outputs (e.g., streamline–surface mappings may be infeasible)

---

## 5. New Ideas Generated

### 5.1 Parcel × Bundle Combinations

- Intersect tractography bundles with atlas parcels
- Produce:
    - Parcel-within-bundle time series

---

### 5.2 “Bundle Time Series” (BT series)

- Extract BOLD signals from white matter bundles
- Potential for:
    - GM–WM correlation matrices

---

### 5.3 Generalized Masking Framework

- Many operations reduce to:
    - Masking + aggregation
- Suggests a unified abstraction

---

### 5.4 Atlas Composition Operations

(primarily for BAT)

- Intersection
- Union
- Outer product of parcels

---

### 5.5 Configurable Transform Pipelines

- Step-based transformations:
    - e.g., square values → union → transform space

---

### 5.6 Graph-Based Transform Engine

- Treat transforms as:
    - Nodes: spaces/formats
    - Edges: transforms
- Compute optimal paths dynamically

---

### 5.7 Multi-modal Integration

- Combine:
    - fMRI (time series)
    - dMRI (bundles)
    - Atlases
- Unified outputs

---

## 6. Practical Next Steps (Agreed)

- Define:
    - CLI interface
    - Config grammar
- Implement:
    - Core BDT functionality first
- Defer:
    - BAT to later (e.g., hackathon)
- Prototype:
    - Using a dataset with multiple derivative sources

---

## Bottom Line

The meeting converged on a **clean architectural decomposition**:

- **BDT = apply atlases to data (core engine)**
- **BAT = construct/modify atlases (advanced preprocessing)**

With a strong emphasis on:

- Explicitness over heuristics
- Config-driven transformations
- Graph-based transform resolution
- Early focus on simple, high-demand use cases

This establishes a scalable foundation without overcommitting to complex atlas algebra in the first iteration.