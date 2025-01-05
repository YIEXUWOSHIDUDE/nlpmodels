README for the Medical Record Classification Project
________________________________________
Project Title
Medical Pre-Diagnosis Department Division Model
________________________________________
Introduction
This project addresses the critical shortage of medical professionals by developing a Medical Pre-Diagnosis Department Division Model using advanced Natural Language Processing (NLP) techniques. The model classifies medical records based on patient-reported symptoms, accurately directing cases to the appropriate medical departments.
By integrating domain-specific large language models (LLMs) and efficient training techniques, the project aims to reduce the workload of healthcare providers and improve the efficiency of medical consultations.
________________________________________
Features
1.	Automatic Department Classification: The model utilizes self-reported symptoms to recommend relevant medical departments.
2.	Resource Efficiency: Incorporates LoRA (Low-Rank Adaptation) for training, enabling deployment on resource-constrained devices.
3.	High Accuracy: Fine-tuned with clinical-standard datasets to achieve excellent classification accuracy.
4.	Multilingual Support: Capable of processing medical records in multiple languages for broader applicability.
5.	Robust Preprocessing: Automated data cleaning and processing for enhanced model performance.
________________________________________
Dataset
1.	Source: Open-source medical dataset, Huatuo26M-Lite, processed as zeng981/nlpdataset (https://huggingface.co/datasets/zeng981/nlpdataset)on Hugging Face.
2.	Details: 
o	Size: Approximately 178,000 entries in Chinese.
o	Columns: Input (symptoms) and Output (medical department).
3.	Preprocessing: 
o	Unnecessary columns (e.g., id, answer) removed.
o	Added an instruction column to guide the model.
o	Renamed columns: question → input, label → output.
o	Data split: 80% training, 20% testing.
________________________________________
Model Development
1.	Architecture:
o	Embedding Layer for text vectorization.
o	Convolutional Neural Network (CNN) for feature extraction.
o	Dropout Layers for reducing overfitting.
o	Dense Layers for classification.
2.	Training:
o	Loss Function: Cross-Entropy Loss for multi-class classification.
o	Optimizer: Adam optimizer for efficient parameter updates.
o	Early Stopping: Monitors validation loss to prevent overfitting.
3.	Evaluation Metrics:
o	Accuracy: Measures correct classifications.
o	F1-Score: Balances precision and recall.
o	ROUGE-1 Score: Captures content overlap in medical context.
Model Resources:
- Hugging Face Model Repository:(https://huggingface.co/zeng981/nlpgemma29B) (https://huggingface.co/zeng981/NLPGLM-4-9B) (https://huggingface.co/zeng981/NLPqwen-1.8b) (https://huggingface.co/zeng981/NLPLlama3.2-3b) (https://huggingface.co/zeng981/NLPQwen2-7b )
________________________________________
Installation
1.	Clone the repository: 
2.	git clone git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
3.	Navigate to the project directory: 
4.	cd LLaMA-Factory
5.	Install dependencies: 
6.	pip install -e ".[torch,metrics]"
________________________________________
Usage
1.	Data Preparation: 
o	Place your dataset in the data/ folder.
o	Update the file paths in the script as needed.
2.	Run Training: 
3.	python train_model.py
4.	Evaluate Model: 
5.	python evaluate_model.py
________________________________________
Future Improvements
1.	Extend dataset to include rare diseases.
2.	Address ethical concerns, including bias and privacy.
3.	Incorporate hybrid models combining rule-based systems with LLMs.
