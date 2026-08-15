# Training And Evaluation
In this phase, we will simultaneously train 3 models, evaluate them, and register the best using a coordinated workflow. We will execute a Domino Flow that trains the three models, evaluate the models using the Domino Experiment Manager, and register the best ones in the Model Registry.

## Exercise Instructions

1. In the workspace, open the terminal

2. Run the Flow

3. Watch the Flow execution

4. Click "Experiment Manager" (main window, left-hand column)

5. Select all runs (3) and click "Compare"

6. Select the one with the best metrics (hint: XGBoost)

7. Review the complete traceability and all factors

8. Click "Register Model From Run" in upper right

9. Create model name

10. Review governance requirements 

This concludes the "Training and Evaluation" section of the workshop.

## New Domino Concepts

**Domino Flows:**
> Domino Flows is a visual workflow orchestration tool that allows users to build, schedule, and monitor complex data pipelines by connecting multiple tasks, jobs, and models in a directed acyclic graph (DAG). This enables teams to productionize end-to-end ML workflows with built-in error handling, dependency management, and automatic retries without writing orchestration code.

**Experiment Manager:**
> Experiment Manager is a centralized tracking system that automatically captures and compares all experiment runs, including parameters, metrics, code versions, and results in a searchable interface. This accelerates model development by enabling data scientists to quickly identify the best-performing models, understand what changes improved performance, and reproduce any past experiment.

**Model Registry:**
> Model Registry is a centralized repository that catalogs all trained models with their metadata, performance metrics, lineage, and deployment status throughout their lifecycle. This provides governance and collaboration capabilities by enabling teams to discover, compare, promote, and deploy models while maintaining full auditability and compliance documentation.
