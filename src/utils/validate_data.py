import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.expectations.core import (
    ExpectColumnToExist,
    ExpectColumnValueLengthsToEqual,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeUnique,
    ExpectColumnValuesToNotBeNull,
)
from typing import List, Tuple


def _build_loan_expectation_suite() -> ExpectationSuite:
    suite = ExpectationSuite(name="loan_default")

    suite.add_expectation(ExpectColumnToExist(column="LoanID"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="LoanID"))
    suite.add_expectation(ExpectColumnValuesToBeUnique(column="LoanID"))
    suite.add_expectation(ExpectColumnValueLengthsToEqual(column="LoanID", value=10))

    for col in [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio",
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner",
        "Default",
    ]:
        suite.add_expectation(ExpectColumnToExist(column=col))

    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="Education",
            value_set=["Bachelor's", "High School", "Master's", "PhD"],
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="EmploymentType",
            value_set=["Full-time", "Part-time", "Self-employed", "Unemployed"],
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="MaritalStatus",
            value_set=["Divorced", "Married", "Single"],
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="HasMortgage", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="HasDependents", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="LoanPurpose",
            value_set=["Auto", "Business", "Education", "Home", "Other"],
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="HasCoSigner", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="Default", value_set=[0, 1])
    )
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="LoanTerm", value_set=[12, 24, 36, 48, 60]
        )
    )

    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="Age", min_value=18, max_value=69)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="Income", min_value=15000, max_value=150000)
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="LoanAmount", min_value=5000, max_value=250000
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="CreditScore", min_value=300, max_value=850
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="MonthsEmployed", min_value=0, max_value=120
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="NumCreditLines", min_value=1, max_value=4
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="InterestRate", min_value=2.0, max_value=25.0
        )
    )
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="DTIRatio", min_value=0.1, max_value=0.9)
    )

    for col in [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio",
        "Default",
    ]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    return suite


def validate_loan_data(df) -> Tuple[bool, List[str]]:
    """
    Data validation for the Loan Default dataset using Great Expectations.

    Validates schema, categorical domains, numeric ranges, and identifiers
    before model training.
    """
    print("Starting data validation with Great Expectations...")

    context = gx.get_context(mode="ephemeral")
    suite = _build_loan_expectation_suite()
    context.suites.add(suite)

    source = context.data_sources.add_pandas("loan_default")
    asset = source.add_dataframe_asset("loan_default")
    batch = asset.get_batch(
        asset.build_batch_request(options={"dataframe": df})
    )
    results = batch.validate(suite)

    failed_expectations = [
        r.expectation_config.type
        for r in results.results
        if not r.success
    ]

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(
            f"Data validation PASSED: {passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"Data validation FAILED: {failed_checks}/{total_checks} checks failed"
        )
        print(f"   Failed expectations: {failed_expectations}")

    return results.success, failed_expectations
