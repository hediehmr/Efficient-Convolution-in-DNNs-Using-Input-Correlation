# Efficient Convolution Implementation: Differential & LSH

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-HPC_Optimization-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Academic_Project-green.svg)]()

## 📌 Project Overview
This repository contains the implementation of advanced convolution techniques aimed at optimizing Deep Neural Networks (DNNs). The focus is on reducing computational complexity (multiplications and additions) using mathematical transformations.

This project was implemented as the final exam for the **"Efficient Implementation of Deep Neural Networks"** course at **Sharif University of Technology**.

## 🚀 Key Implementations
We implemented and analyzed two specific optimization techniques from scratch (without relying on high-level framework optimizations):

1.  **Differential Convolution:**
    * Reduces redundancy by processing the *difference* between input elements rather than raw values.
    * **Goal:** Minimize the number of multiplications in the convolution operation.
    * **Complexity Analysis:** Detailed breakdown of MAC (Multiply-Accumulate) operations provided in the report.

2.  **Locally Sensitive Hashing (LSH) for Convolution:**
    * Uses hashing to group similar input vectors (windows) and kernels.
    * **Goal:** Approximate convolution by computing results for "buckets" rather than individual sliding windows, significantly speeding up inference at the cost of negligible accuracy drop.

## 🛠️ Mathematical Foundation
The implementation relies on minimizing the operation count. For example, in the LSH method, the complexity is broken down into:
* **A:** Grouping overhead in the initial phase.
* **B:** Window grouping overhead per input image.
* **C:** The core convolution operation on reduced groups.
* **Total Multiplications:** $A \times B \times C$ (optimized vs. standard $O(N^2)$).

## 📄 Full Documentation
For a deep dive into details, please refer to the document below:

👉 **[Download Full Project Report (PDF)](Final_Report_Persian.pdf)**
