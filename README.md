# credit-risk-model

## Credit Scoring Business Understanding
Credit risk refers to the risk that a borrower or counterparty will fail to meet its contractual financial obligations, resulting in a potential financial loss to the lender.
### Basel II and the Need for Model Interpretability and Documentation

* According to regulatory guidance referenced in the World Bank Credit Scoring Approaches Guidelines, a model is a quantitative method or system that transforms input data into estimates used for financial decision-making, and reliance on such models inherently introduces model risk.

* The Basel II Capital Accord places strong emphasis on accurate risk measurement, transparency, and regulatory oversight in credit risk modeling. Under Basel II, banks are required to justify how credit risk models estimate the probability of default (PD) and how these estimates influence capital allocation. This regulatory environment makes model interpretability, traceability, and documentation essential, not optional.

* An interpretable model allows risk managers, auditors, and regulators to understand why a borrower is classified as high or low risk, how input variables affect predictions, and whether the model’s behavior aligns with economic intuition. Well-documented models also support validation, stress testing, and ongoing monitoring, which are explicit expectations under Basel II’s Internal Ratings-Based (IRB) approaches. Regulatory guidance further treats models themselves as a source of risk, requiring strong governance across the entire model lifecycle.

### Use of a Proxy Default Variable and Associated Business Risks

Creating a proxy variable is necessary to enable supervised learning and to approximate real-world credit risk outcomes in the absence of a direct default label. However, this approach introduces business risks:

* The proxy may not fully capture true default behavior, leading to label noise.
* Model predictions may reflect operational or reporting artifacts rather than actual borrower creditworthiness.
* Decisions based on an imperfect proxy can result in mispricing of risk, inappropriate credit approvals or rejections, and potential regulatory scrutiny.

Therefore, proxy construction must be clearly justified, conservative in nature, and thoroughly documented so that stakeholders understand its limitations. This approach is consistent with regulatory expectations that model assumptions and weaknesses be transparent and subject to ongoing review.

### Trade-offs Between Interpretable and Complex Models in a Regulated Context

There is a fundamental trade-off between model interpretability and predictive performance in credit risk modeling:

Simple, interpretable models (e.g., Logistic Regression with Weight of Evidence encoding):
* Easy to explain to regulators and business stakeholders
* Stable over time and easier to monitor
* Align well with Basel II governance and validation expectations
* May sacrifice some predictive accuracy

Complex, high-performance models (e.g., Gradient Boosting, ensemble methods):
* Often deliver higher predictive power and better risk separation
* Harder to explain, validate, and defend in regulatory reviews
* Require advanced explainability techniques and stronger governance controls
* Increase model risk and operational complexity

In regulated financial environments, institutions often prioritize interpretability, robustness, and compliance over marginal gains in performance, as increased model complexity materially raises governance, validation, and model risk management requirements. As a result, simpler models are frequently preferred for production credit decisioning, while complex models may be used as challenger models or for internal risk insights.

### Credit Risk EDA
#### Project Structure

- `data/`
Contains the raw and processed data files (ignored in git).

- `notebooks/`
Jupyter notebooks for interactive exploration and analysis.
Key notebook: `eda.ipynb` — contains the main exploratory data analysis workflow.

- `src/`
Placeholder for Python modules for data processing and future model development.

- `tests/`
Unit tests for src/ modules, ensuring correctness of data processing and EDA functions.

- `.github/workflows/ci.yml`
CI/CD workflow for testing and style checks.

#### How to use
* Clone the repo and install dependencies from `requirements.txt`.
* Open `notebooks/eda.ipynb` to explore the dataset and run EDA analyses.

#### Requirements
* See `requirements.txt` for the necessary Python libraries.
