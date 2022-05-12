# Assisting Physicians with Automated Report Generation for X-ray images
## Course Project: CS 769 - Spring 2022

With the recent advancements in Natural Language Processing, analyzing free-form text to extract semantic information has become significantly easier. At the same time, these advances have also led to the rise in the popularity of text generation systems such as GPT-3 that can generate free-form text given a prompt. Researchers have investigated applying these advances to assist the medical community with tasks such as report generation for medical images. However, the majority of the work has focused on caption generation, which does not produce full-fledged reports. In this work, we propose to learn joint aligned embeddings from x-ray images and their reports and train a decoder to generate reports from this joint space. Due to different physicians' report writing styles, a major challenge is generating reports with a uniform style. To counter this, we propose to generate reports in steps, starting by defining a fixed structure for the reports and then using these structures as intermediate representations to generate a complete report. Additionally, we re-implement a method for learning a vision-language aligned representation space and explore its efficacy given normal data augmentation techniques.

## Usage:
- To train and run report generation: 
`python3 main.py --report_gen True`
- To perform joint multi-modal training (aligning image encoder and text encoder): `python3 main.py --pretraining True`
- To perform mlm pretraining for the text encoder: `python3 main.py --mlm_pretraining True`

## Information Regarding Relevant Paths
Using the input parameter `local`, the code first determines the base directory (`dir_base`). Then, it expects models and files in the following paths:
- The dataframe consisting reports, image encoder models and dataset under `os.path.join(dir_base, 'CS769_Gloria_models').
- The text encoder under `os.path.join(dir_base,'bert_base_uncased')`
- 