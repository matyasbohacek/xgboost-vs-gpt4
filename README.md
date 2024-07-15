![cover](https://github.com/user-attachments/assets/6369bf7b-3778-4cd2-8bbe-ae7de1308bf7)


# When XGBoost Outperforms GPT-4 on Text Classification: A Case Study

<p style="color:#808080;"><i>published at <a href="https://trustnlpworkshop.github.io">NAACL 2024 Workshop on Trustworthy Natural Language Processing (TrustNLP 2024)</a></i></p>

[See paper](https://aclanthology.org/2024.trustnlp-1.5/) — [See poster]([https://aclanthology.org/2024.trustnlp-1.5/](https://drive.google.com/file/d/1A77T_RBLkHWnpcgcAE5W6Cr3KfX705n1/view?usp=sharing)) — [Contact us](mailto:maty-at-stanford-dot-edu)

Large language models (LLMs) are increasingly used for applications beyond text generation, ranging from text summarization to instruction following. One popular example of exploiting LLMs’ zero- and few-shot capabilities is the task of text classification. This short paper compares two popular LLM-based classification pipelines (GPT-4 and LLAMA 2) to a popular pre-LLM-era classification pipeline on the task of news trustworthiness classification, focusing on performance, training, and deployment requirements. We find that, in this case, the pre-LLM-era ensemble pipeline outperforms the two popular LLM pipelines while being orders of magnitude smaller in parameter size.

## Getting Started

Set up a Python environment (Python 3.8 is recommended) and install the dependencies using `pip install -r requirements.txt`.

### Reproducing Paper Results

The `experiments/pipeline_extraction.py` and `experiments/gpt_extraction.py` scripts contain two different approaches to the news trusthworthiness classification task described in the paper.

To reproduce the results in our paper, download the [Verifee dataset](https://forms.gle/3HZ5fn6Mi8rQEApx9), and set up the according paths.

### Using the Methodology for Custom Projects

To adapt the methodology to different tasks and datasets:

- update the feature composition of the ensemble pipeline;
- gather new feature models for the respective features;
- and update the prompt in the GPT pipeline.

To go beyond the specific ensemble and LLM architectures we chose (Electra and GPT-4, respectively):

- update the Hugging Face Transformers pipeline in the ensemble pipeline;
- update the LLM interfacing function (e.g., call a different API).

## Citation

```bibtex
@inproceedings{bohacek-bravansky-2024-xgboost,
    title = "When {XGB}oost Outperforms {GPT}-4 on Text Classification: A Case Study",
    author = "Bohacek, Matyas  and
      Bravansky, Michal",
    booktitle = "Proceedings of the 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.trustnlp-1.5",
    pages = "51--60"
}
```

## Remarks & Updates

- (**July 14, 2024**) Note that OpenAI's API changes the interfaced model regularly, and thus the precise results you will observe might, to some extent, differ from those reported in the paper. Our data was collected in August 2023.
