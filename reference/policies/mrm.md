# MODEL RISK MANAGEMENT

```
classification:
  rule:
  artifacts:
    - model-risk
approvers:
    - model-gov-org
stages:
  - name: Business Case Development
    evidenceSet:
      - id: Local.business-case
        name: Business Case
        description: Define the business problem and initial risk assessment
        definition:
          - artifactType: input
            details:
              label: What is the purpose of the model?
              type: textarea
          - artifactType: input
            details:
              label: What is the type of the business case?
              type: select
              options:
                - Model development (incl new model)
                - Model change
          - artifactType: input
            details:
              label: What are expected business benefits for this change or development?
              type: textarea
    approvals:
      - name: Business Case Sign Off
        allowAdditionalApprovers: true
        approvers:
          - model-gov-org
        evidence:
          id: Local.business-case-signoff
          name: Business Case Signoff
          description: Review and approve the business case
          definition:
          - artifactType: input
            details:
              label: Who is allowed to perform a sign-off of this model change request?
              type: select
              options:
                - Model owner or model approver
                - Model approver
          - artifactType: input
            details:
              label: Did you read the business case?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Have you discussed the business case with the model owner?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            aliasForClassification: model-risk
            details:
              label: Do you classify this model as being High or Low risk ?
              type: radio
              options:
                - High
                - Low
          - artifactType: input
            details:
              label: Sign-off date
              type: date
  - name: Requirement gathering
    evidenceSet:
      - id: Local.define-requirement
        name: Define Requirement
        description: Describe the requirement
        definition:
          - artifactType: input
            details:
              label: List all the requirements
              type: textarea
      - id: Local.change-plan
        name: Define the change plan
        description: Describe the change plan
        definition:
          - artifactType: input
            details:
              label: Describe the change plan
              type: textarea
        visibilityRule: classificationValue == "High"
  - name: Model development
    evidenceSet:
      - id: Local.model-building
        name: Model Building
        description: Describe the model building
        definition:
          - artifactType: input
            details:
              label: Describe the model building
              type: textarea
      - id: Local.acceptance-testing
        name: Acceptance Testing
        description: Describe the acceptance testing
        definition:
          - artifactType: input
            details:
              label: Describe the acceptance testing
              type: textarea
  - name: User Acceptance Testing
    evidenceSet:
      - id: Local.user-acceptance-test
        name: User Acceptance Test
        description: Perform user acceptance testing
        definition:
          - artifactType: input
            details:
              label: What is the model version that will be tested?
              type: textarea
          - artifactType: input
            details:
              label: What is the location of the test model?
              type: textarea
          - artifactType: input
            details:
              label: When is the test model version received?
              type: date
          - artifactType: input
            details:
              label: Is the test model version delivered by the appropriate model developer?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: What is the test model conclusion?
              type: textarea
          - artifactType: input
            details:
              label: Is the corresponding test model documentation received?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: What is the location of the test model documentation?
              type: textarea
          - artifactType: input
            details:
              label: Does the model documentation contain the agreed upon test reports?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Does the test model documentation contain a list of findings and test conclusion and identified weaknesses?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Who is the model developer point of contact for user acceptance feedback?
              type: textarea
          - artifactType: input
            details:
              label: Can the user acceptance test start in your opinion?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Have you informed the model developer about the current state?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: What is the overall conclusion of the user acceptance test?
              type: textarea
          - artifactType: input
            details:
              label: Did the test model pass the user acceptance test?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: If it passed the user acceptance test, why did it pass?
              type: textarea
          - artifactType: input
            details:
              label: If it did not pass the user acceptance test, why did it not pass?
              type: textarea
          - artifactType: input
            details:
              label: In case another iteration is needed, what does the model developer need to do?
              type: textarea
          - artifactType: input
            details:
              label: In case another iteration is needed, when can the model developer make changes?
              type: textarea
          - artifactType: input
            details:
              label: Is this reflected in the model version name?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Is this reflected in the model documentation?
              type: radio
              options:
                - Yes
                - No
  - name: Validation
    approvals:
      - name: Validation sign off
        approvers:
          - model-gov-org
        evidence:
          id: Local.validation-approval-body
          name: Sign-off
          description: The checklist for approvals
          definition:
            - artifactType: input
              details:
                label: Have you read the model validation reports?
                type: radio
                options:
                  - label: 'Yes'
                    value: 'Yes'
                  - label: 'No'
                    value: 'No'
            - artifactType: input
              details:
                label: Have you checked the explainability report?
                type: radio
                options:
                  - label: 'Yes'
                    value: 'Yes'
                  - label: 'No'
                    value: 'No'
            - artifactType: input
              details:
                label: Do you sign off this model?
                type: radio
                options:
                  - label: 'Yes'
                    value: 'Yes'
                  - label: 'No'
                    value: 'No'
    evidenceSet:
      - id: Local.validation_method
        name: Method Validation
        description: Validate the modelling method
        definition:
          - artifactType: input
            details:
              label: Which methodology is the model based on?
              type: textarea
          - artifactType: input
            details:
              label: Is the methodology fit for purpose and are all the methodology choices properly argumented?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Are the input and output data complete, accurate, relevant and consistent?
              type: radio
              options:
                - Yes
                - No
          - artifactType: input
            details:
              label: Did the model owner perform any explainability analysis?
              type: radio
              options:
                - Yes
                - No
      - id: Local.model-quality
        name: Model Quality
        description: Describe the model quality
        definition:
          - artifactType: metadata
            details:
              type: modelmetric
              metrics:
                - name: Accuracy
                  threshold:
                    operator: '>='
                    value: 0.8
                - name: Precision
                  threshold:
                    operator: '=>'
                    value: 0.75
          - artifactType: policyScriptedCheck
            details: {}
      - id: Local.validation_data
        name: Validation Documentation
        description: Validate the data used for the model
        definition:
          - artifactType: input
            details:
              label: Is the model validation completed?
              type: radio
              options:
                - Yes
                - No
          - artifactType: metadata
            details:
              label: Upload model validation report.
              type: file
  - name: Deployment
    evidenceSet:
      - id: Local.monitoring-plan
        name: Monitoring Plan
        description: Describe the monitoring plan
        definition:
          - artifactType: input
            details:
              label: Describe the monitoring plan
              type: textarea
      - id: Local.use-of-model
        name: Use of Model
        description: Describe the use of the model
        definition:
          - artifactType: input
            details:
              label: Describe the use of the model
              type: textarea
```