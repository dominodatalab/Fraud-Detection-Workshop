# NIST AI RISK MANAGEMENT FRAMEWORK
```
classification:
  rule: 
  artifacts:
    - model-risk
stages:
  - name: Problem Definition and Planning
    evidenceSet:
      - id: Local.guidance
        name: Planning Guidance
        description: Guidance for problem definition and planning
        definition:
          - artifactType: guidance
            details:
              text: >-
                This stage involves defining the business problem and conducting initial risk assessment.
                The complexity and risk level of the project will determine the specific process to follow.
                [Learn More](https://www.nist.gov/itl/ai-risk-management-framework)
      - id: Local.business-problem
        name: Model Purpose Document
        description: Defines business problem and model purpose
        definition:
          - artifactType: input
            details:
              label: Provide a detailed description of the business problem and model purpose
              type: textarea
      - id: Local.initial-risk-assessment
        name: Initial Risk Assessment
        description: Conduct initial risk assessment
        definition:
          - artifactType: input
            details:
              label: Document initial risk assessment findings
              type: textarea
      - id: Local.intended-datasets
        name: Intended Data Sources
        description: List of data sources intended for use in the project
        definition:
          - artifactType: input
            details:
              label: Provide a comprehensive list of all data sources intended to be used in this project
              type: textarea
      - id: Local.project-kpis
        name: Project KPIs and Target Metrics
        description: Define the key performance indicators and target metrics for project success
        definition:
          - artifactType: input
            details:
              label: Specify the KPIs and target metrics that will be used to track the success of the project
              type: textarea
    approvals:
      - name: Problem Definition Sign Off
        allowAdditionalApprovers: true
        approvers:
          - andrea_lowe
        evidence:
          id: Local.problem-definition-signoff
          name: Problem Definition Approval
          description: Review and approve the problem definition, risk assessment, and KPIs
          definition:
            - artifactType: input
              details:
                label: Did you review and approve the problem definition, risk assessment, and intended data sources?
                type: radio
                options:
                  - Yes
                  - No
            - artifactType: input
              details:
                label: Do you authorize the project to proceed to the business approval stage?
                type: radio
                options:
                  - Yes
                  - No
  - name: Business Approval
    evidenceSet:
      - id: Local.business-approval-1
        name: Business Approval Sign-off
        description: Checklist for business approval before proceeding to experimentation
        definition:
          - artifactType: input
            details:
              label: Have you reviewed the business problem definition, initial risk assessment, and KPIs?
              type: radio
              options:
                - Yes
                - No
      - id: Local.business-approval-2
        name: Proceed to Experimentation Approval
        description: Authorization to proceed to experimentation
        definition:
          - artifactType: input
            details:
              label: Do you authorize the project to proceed to the experimentation phase?
              type: radio
              options:
                - Yes
                - No
    approvals:
      - name: Business Approval Sign Off
        allowAdditionalApprovers: true
        approvers:
          - andrea_lowe
        evidence:
          id: Local.business-approval-signoff
          name: Business Approval Signoff Evidence
          description: Sign-off evidence for business approval
          definition:
            - artifactType: input
              details:
                label: Did you sign off on the business approval?
                type: radio
                options:
                  - Yes
                  - No
  - name: Experimentation
    evidenceSet:
      - id: Local.experimentation-results
        name: Experimentation Results
        description: Document the results of various techniques and approaches tested
        definition:
          - artifactType: input
            details:
              label: Provide a comprehensive report of all techniques tested, their results, and justification for the selected approach
              type: textarea
      - id: Local.selected-approach
        name: Selected Approach Justification
        description: Justification for the chosen modeling approach
        definition:
          - artifactType: input
            details:
              label: Explain why the selected technique or strategy was chosen over alternatives
              type: textarea
      - id: Local.kpi-progress
        name: KPI Progress Report
        description: Report on the progress towards the defined KPIs and target metrics
        definition:
          - artifactType: input
            details:
              label: Provide an update on the progress towards the project's KPIs and target metrics based on experimentation results
              type: textarea
    approvals:
      - name: Experimentation Sign Off
        allowAdditionalApprovers: true
        approvers:
          - andrea_lowe
        evidence:
          id: Local.experimentation-signoff
          name: Experimentation Signoff
          description: Review and approve the experimentation results
          definition:
            - artifactType: input
              details:
                label: Did you review the experimentation results?
                type: radio
                options:
                  - Yes
                  - No
  - name: Model Design and Development
    evidenceSet:
      - id: Local.model-architecture
        name: Model Architecture Document
        description: Define model architecture and algorithms
        definition:
          - artifactType: input
            details:
              label: Describe the chosen model architecture and algorithms
              type: textarea
      - id: Local.model-limitations
        name: Model Limitations Document
        description: Document model limitations and constraints
        definition:
          - artifactType: input
            details:
              label: List known limitations and constraints of the chosen model
              type: textarea
      - id: Local.model-implementation
        name: Model Implementation Code
        description: Implement model in code
        definition:
          - artifactType: input
            details:
              label: Provide the codebase or implementation details for the model
              type: textarea
      - id: Local.final-results
        name: Final Model Results
        description: Document final model results from the experimentation phase
        definition:
          - artifactType: input
            details:
              label: Provide comprehensive results of the final model, including performance metrics and comparison to project KPIs
              type: textarea
    approvals:
      - name: Model Design and Development Sign Off
        allowAdditionalApprovers: true
        approvers:
          - andrea_lowe
        evidence:
          id: Local.model-design-signoff
          name: Model Design Signoff
          description: Review and approve the model design and development
          definition:
            - artifactType: input
              details:
                label: Did you approve the model design and development?
                type: radio
                options:
                  - Yes
                  - No
  - name: Deployment Planning and Architectural Review
    evidenceSet:
      - id: Local.deployment-strategy
        name: Deployment Strategy Document
        description: Define deployment strategy and monitoring plan
        definition:
          - artifactType: input
            details:
              label: Outline the deployment strategy, including infrastructure and monitoring plans
              type: textarea
      - id: Local.scalability-plan
        name: Scalability and Performance Plan
        description: Document the plan for system scalability and performance
        definition:
          - artifactType: input
            details:
              label: Describe how the system will handle increased load and maintain performance
              type: textarea
      - id: Local.security-measures
        name: Security and Data Protection Measures
        description: Outline security measures and data protection protocols
        definition:
          - artifactType: input
            details:
              label: Detail the security measures and data protection protocols in place
              type: textarea
    approvals:
      - name: Deployment Planning Sign Off
        allowAdditionalApprovers: true
        approvers:
          - andrea_lowe
        evidence:
          id: Local.deployment-signoff
          name: Deployment Planning Signoff
          description: Review and approve the deployment strategy
          definition:
            - artifactType: input
              details:
                label: Did you approve the deployment strategy?
                type: radio
                options:
                  - Yes
                  - No
```