# Credit Risk Prediction for Buy-Now-Pay-Later Lending

## Project Overview

This project aims to build a **credit scoring system** for Bati Bank to support a **buy-now-pay-later (BNPL)** partnership with an eCommerce company. The goal is to evaluate the risk associated with lending to new customers using their behavioral transaction data. Since traditional credit history may not be available, we rely on alternative features and proxy variables to infer creditworthiness.

We follow an end-to-end machine learning approach — including **data preprocessing**, **proxy label creation**, **feature engineering**, **model training**, and **deployment** — all while ensuring regulatory compliance, interpretability, and model robustness.

---

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable Models

The **Basel II Accord**, particularly under the Internal Ratings-Based (IRB) approach, allows banks to use internal models to estimate risk parameters such as Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). However, this flexibility comes with a strict requirement: **risk models must be interpretable and auditable**.

Interpretability is essential for meeting regulatory expectations. Supervisors must understand how risk assessments are made to ensure models are not manipulated to reduce capital requirements. Transparent models support **validation**, uncover **bias or errors**, and ensure the model behaves reliably under stress conditions.

Moreover, models that can be understood foster **trust** among regulators, internal risk teams, and even customers. They also allow institutions to conduct **scenario analysis** and **stress testing**, which are vital in financial risk management. In essence, interpretability is not just a technical feature — it is a regulatory necessity under Basel II, ensuring that risk decisions are fair, explainable, and accountable.

---

### 2. Creating a Proxy for Default: Necessity and Risk

In this project, we lack a direct label for whether a customer defaulted. To address this, we use a **proxy target variable** derived from customer behavior — particularly **Recency, Frequency, and Monetary (RFM)** transaction metrics. Customers with low activity levels and spending are clustered and labeled as "high-risk," serving as stand-ins for defaulted customers.

This approach, while practical, introduces several challenges. A proxy is an **approximation**, not ground truth. For instance, a customer might stop purchasing for personal or seasonal reasons, not because of credit issues. If the proxy label is inaccurate, the model may learn **false patterns**, resulting in poor real-world predictions.

The use of proxies can also introduce **biases**, especially if behavioral patterns correlate with sensitive attributes like region or income level. This could lead to **unfair lending decisions** if not carefully managed. Furthermore, model performance may fluctuate significantly depending on how the proxy is constructed, leading to **model instability**.

Despite these risks, proxy variables are often necessary in data-limited environments. What’s critical is that they are created using business logic, are validated iteratively, and are revisited regularly as more real-world data becomes available.

---

### 3. Balancing Interpretability and Predictive Power

Building a credit scoring model requires a careful balance between **performance and transparency**. In regulated environments like banking, this trade-off is particularly crucial.

**Simple models** — such as Logistic Regression, especially when combined with Weight of Evidence (WoE) encoding — are favored for their clarity. They allow stakeholders to understand how each input contributes to risk scores, making them suitable for regulatory review, compliance audits, and internal validation.

However, simple models may not capture complex interactions within the data, potentially leading to lower accuracy. On the other hand, complex models like Gradient Boosting or Neural Networks can deliver superior predictive performance, especially on rich and high-dimensional datasets. But their black-box nature makes it difficult to justify predictions, posing risks in regulated settings.

In this context, the best approach may involve leveraging complex models alongside explainability tools such as SHAP or LIME. These tools provide local and global explanations of model behavior, allowing teams to maintain high performance while still adhering to transparency requirements.

Ultimately, the choice of model architecture must reflect both business priorities and regulatory obligations. The aim is not only to make accurate predictions, but to ensure those predictions can be trusted, explained, and defended.

