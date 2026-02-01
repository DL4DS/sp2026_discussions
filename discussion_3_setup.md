# Discussion 3: SCC

SCC dashboard: https://scc-ondemand1.bu.edu/pun/sys/dashboard/ 

### 0. Download the `discussion_3.ipynb` from the link below

Link: https://github.com/DL4DS/fa2026_discussions/blob/main/discussion_3.ipynb 

### 1. Create a session on SCC

- Access `/projectnb/dl4ds/student/`  
    - Click “Files” and select `/projectnb/dl4ds/student/your_user_name` 
    - Upload `discussion_3.ipynb` from GitHub to the SCC folder 

- Create a session 
    - Click “Interactive apps” - "Jupyter Notebook" 

    - List of modules to load (space separated): 
        - select `miniconda` and `academic-ml/spring-2026` modules 

    - Pre-Launch Command (optional): 
        - fill `conda activate spring-2026-pyt` 
        - if you are requesting a TensorFlow environment, fill `conda activate spring-2026-tf` 

    - Interface: 
        - choose `lab ` 

    - Working Directory: 
        - select `/projectnb/dl4ds/students/your_user_name` 

    - Number of hours: `2` 

    - Number of cores: `1` 

    - Number of gpus: `1` 

    - GPU compute capability: `3.5` (you can choose 6.0 or higher for your other tasks) 

    - Projects: `dl4ds` 

### 2. Implement a 10k * 10k Matrix Multiplication on GPU

### 3. Import the dataset from shared folder and run a shallow network for training

```python
df = pd.read_csv('/projectnb/dl4ds/materials/diabetes.csv')
```

The dataset is the same one from your `hw3` question 2-3, this step is to tell how you can read files from shared  folders. For example, for your group projects, you can upload datasets under `/projectnb/dl4ds/projects/` directory.