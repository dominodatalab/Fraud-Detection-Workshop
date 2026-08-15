# Data Engineering
In this phase, we will continue to prepare the data for training by executing some simple Data Engineering tasks. We will be executing a Domino job that reads the updated CSV from a Domino Dataset, performs simple feature engineering such as normalization of the data, adding a derived column, saving the data, and taking a snapshot of the data.

## Exercise Instructions

1. Within the workspace, open and review the `data_engineering.py` script (do not modify)

2. With all code committed, click "Run Job"

3. Fill out the form as shown below:

4. Back on the main tab, in the project click "Jobs" on the left

5. Expand the right for details
   - Notice how everything updates itself in real time

6. Review the artifacts created by the job run

This concludes the "Data Engineering" section of the workshop.

## New Domino Concepts

**Jobs:**
> Jobs are scheduled or on-demand executions of scripts, notebooks, or applications that run asynchronously in Domino without requiring an interactive session. They enable automated workflows, batch processing, and production deployment of models while providing full reproducibility, monitoring, and resource management capabilities.

**Domino Dataset Snapshots:**
> Dataset Snapshots are immutable, point-in-time versions of Domino Datasets that capture the exact state of data at a specific moment. This enables perfect reproducibility of experiments and models by guaranteeing that the same data version can be accessed months or years later, while also supporting compliance and audit requirements.

**Artifacts:**
> Artifacts are files or outputs generated during project executions (such as models, reports, or visualizations) that Domino automatically tracks and versions alongside the code and environment that produced them. This provides complete lineage and traceability of all project outputs, making it easy to reproduce results, share deliverables with stakeholders, and maintain governance over model assets.


