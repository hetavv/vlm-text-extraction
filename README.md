# vlm-text-extraction
# VLM Assignment - MLE

This repository contains the code and results for the Vision Language Model (VLM) assignment, focusing on gujarati text extraction from scanned documents, advanced image processing, and VLM fine-tuning.

The primary Multi-modal Model used for this project is the **Google Gemma 3-12b-it** multimodal model, available on [Hugging Face](https://huggingface.co/google/gemma-3-12b-it)

## Repository Structure

This submission is organized into three main Google Colab notebooks for clarity and reproducibility:

1.  `data_prep_and_image_enhancement.ipynb`: Handles dataset creation, ground truth generation, and advanced image processing using a GAN for image restoration.
2.  `baseline_inference.ipynb`: Performs baseline text extraction using an off-the-shelf VLM and evaluates its performance on both raw and enhanced scans.
3.  `model_fine_tuning.ipynb`: Demonstrates the fine-tuning process of the chosen VLM using parameter-efficient methods.

All the files and notebooks that are present in this repo, along with all the raw, enhanced images, ground truths and Finetuned gemma lora adapter checkpoints are also present here in this [Drive link](https://drive.google.com/drive/folders/1xSC6Ys7WwtskumzbtIagm1jePIIkCXZF?usp=sharing)

### 1. Data Preparation and Image Enhancement

**Notebook:** `data_prep_and_image_enhancement.ipynb` : addresses Task 1 (Dataset) and Task 2 (Advanced Image Processing) of the assignment


This notebook covers the foundational steps of the project: dataset preparation, robust ground truth generation, and advanced image preprocessing.

* **Dataset Structuring:** The raw dataset was systematically split into 70% training, 15% validation, and 15% test sets, organized into a JSON file format for efficient access.

* **Data Integrity & Ground Truth Analysis:**
    * Initial OCR (Tesseract) generated ground truth for Gujarati text extraction.
    * Manual inspection revealed critical issues: noise interference (markings, handwritten notes), OCR misinterpretations (dots as digits, barcodes as noise), and errors from improper scans.
    * A robust refinement process was implemented using regular expressions and LLM (GPT) assistance to correct and clean the ground truth.
    * Cleaned ground truth is stored separately (`ground_truth_cleaned`) maintaining the train/test/validation structure, enabling clear comparison with original OCR output
    * <img width="1286" alt="Screenshot 2025-06-01 at 11 22 18 PM" src="https://github.com/user-attachments/assets/1b5f9ef7-0bec-4be2-a6b0-b9dd3b9e8046" />



* **Advanced Image Preprocessing:**
    * After trying out various diffusion and GAN techniques, the **DocRes model** was selected for image restoration as it was performing the best for task at hand. DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks [[Arxiv](https://arxiv.org/abs/2405.04408)]
    * This choice was based on its effectiveness in mitigating common artifacts in scanned documents, such as background noise, text-bleed, stains, and aging effects, which directly impacts downstream VLM performance.
    * ![WhatsApp Image 2025-05-31 at 15 02 13](https://github.com/user-attachments/assets/5b4fac44-9865-4f85-ad79-837cb3d0a010)
    * <img width="1203" alt="Screenshot 2025-06-01 at 11 12 53 PM" src="https://github.com/user-attachments/assets/abb8bd2a-f49d-4982-93b6-9a6753616be3" />

**Notebook:** `baseline_inference.ipynb` : addresses task 3 (baseline inference), task 4 (Text organization), and task 6 (comparison)

This notebook addresses the baseline performance evaluation of the chosen Vision-Language Model for text extraction.

* **Model Application:** The **Google Gemma 3-12b-it** model was used for text extraction. Due to compute constraints, inference was performed on both raw source images and DocRes-cleaned images in chunks of 25.
* **Prompting Strategy:** Text was extracted using the following prompt:

    ```python
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": "Please extract and transcribe all the text you can see in this image. The text is in Gujarati language. Just provide the text content without any additional commentary"}
            ]
        }
    ]
    ```

* **Comprehensive Evaluation:** Performance was rigorously evaluated using the following metrics, calculated after normalizing the text:
    * **Word Error Rate (WER)**
    * **Character Error Rate (CER)**
    * **Custom Sequence-Accuracy Score:** This metric currently calculates a basic line-match accuracy, where reference and hypothesis texts are split into lines and stripped for comparison. Further development is needed to prioritize specific aspects like correct sentence structure or text positioning based on project requirements.

#### Baseline Evaluation Results

| Metric                             | Raw Scans | Enhanced Scans | Delta (Enhanced - Raw) |
| :--------------------------------- | :-------- | :------------- | :--------------------- |
| Word Error Rate (WER)              | 0.6476    | 0.6990         | 0.0514                 |
| Character Error Rate (CER)         | 0.4311    | 0.4742         | 0.0431                 |
| Custom Sequence-Accuracy Score     | 0.0184    | 0.0133         | -0.0051                |

---

## 3. Model Fine-Tuning

**Notebook:** `model_fine_tuning.ipynb`

This notebook details the fine-tuning process for the **Google Gemma 3-12b-it** Vision-Language Model.

* **Fine-Tuning Data:** The model was fine-tuned using the 70 images designated for the training set, leveraging the carefully curated ground truth.
* **Parameter-Efficient Tuning:** Parameter-efficient tuning methods, such as LoRA (Low-Rank Adaptation), were employed to optimize performance while significantly reducing computational cost and memory footprint, making the fine-tuning process more efficient.
* **Loss Reduction:** The fine-tuning process successfully demonstrated a reduction in loss, indicating improved model performance on the specific dataset.
* <img width="703" alt="Screenshot 2025-06-01 at 11 53 20 PM" src="https://github.com/user-attachments/assets/448a6d31-ebad-4866-8629-5c8adb1753a0" />


---

## Comparison

While the assignment aimed to compare the baseline (pre-trained) model's performance with the fine-tuned model, <ins> full execution of this comparison was not completed due to computational resource constraints </ins>. Attempts and preliminary experimental notebooks for this phase are provided in the `fine_tuning_experiments/` folder.

## Deliverables & Project Assets

* All end-to-end reproducible code is provided as Python notebooks in this github repo - please reload incase the notebooks don't render
* The `dataset_splits.json` file, detailing the training, validation, and test splits, is included.
* All folders with images, Doc-Res improved images, ground truths, cleaned up ground truths and Finetuned gemma lora adapter checkpoints are available in [Drive here](https://drive.google.com/drive/folders/1xSC6Ys7WwtskumzbtIagm1jePIIkCXZF?usp=sharing).
* Experimental notebooks related to fine-tuning evaluation attempts are located in the `fine_tuning_experiments/` folder.

---
