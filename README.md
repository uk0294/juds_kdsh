# Conference Assignment System

## Overview
This project implements a system for assigning unlabelled research papers to specific academic conferences. The assignment is based on the similarity between the embeddings of unlabelled papers and labeled papers with known conference metadata. Additionally, the system uses an LLM (Language Model) to generate rationales for the classification.

### Key Features:
- Embedding-based similarity matching using **Pathway**.
- LLM-assisted classification and rationale generation using **OpenAI GPT-4**.
- Modular architecture for easy maintenance and extension.
- Saves results, including classification and rationale, to a CSV file.

---

## Directory Structure
```
conference_assignment/
│
├── main.py                   # Main script to run the program
├── utils/
│   ├── __init__.py           # Makes `utils` a Python package
│   ├── pathway_utils.py      # Pathway-related utilities
│   ├── llm_utils.py          # LLM interaction utilities
│   └── io_utils.py           # Input/output utilities
└── data/
    └── results.csv           # Output file for classification results
```

---

## Requirements
### Software
- Python 3.8+
- OpenAI API Key (for GPT-4).

### Python Libraries
The project uses the following libraries:
- **Pathway**: For handling embeddings and vector stores.
- **LangChain**: For interacting with the LLM.
- **PyPDF2**: For parsing PDF files.
- **dotenv**: For managing environment variables.
- **csv**: For handling CSV input/output.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conference-assignment.git
   cd conference-assignment
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

---

## Usage
1. **Prepare Data Sources**:
   - Ensure that the labeled and unlabelled datasets are available on Google Drive.
   - Update the `object_id` fields in `main.py` with the appropriate Google Drive file IDs for:
     - Labeled data
     - Unlabelled data

2. **Run the Program**:
   Execute the main script:
   ```bash
   python main.py
   ```

3. **Output**:
   - The classified results, along with the rationale, will be saved to `data/results.csv`.

---

## Configuration
### File: `main.py`
- **Google Drive Object IDs**:
  - Replace the placeholders for labeled and unlabelled data:
    ```python
    object_id="your_labeled_data_id"      # For labeled data
    object_id="your_unlabeled_data_id"    # For unlabelled data
    ```

- **Output File**:
  - Update the `OUTPUT_FILE` variable to set a different output file name:
    ```python
    OUTPUT_FILE = "data/your_output_file.csv"
    ```

- **LLM Configuration**:
  - Modify the LLM model or temperature settings if needed:
    ```python
    llm = OpenAI(model="gpt-4", temperature=0)
    ```

---

## Results
The output CSV file contains the following columns:
- **Paper ID**: The ID of the unlabelled paper.
- **Publishable**: A binary value indicating whether the paper is assigned to a conference (`1`) or not (`0`).
- **Conference**: The assigned conference.
- **Rationale**: The LLM-generated explanation for the assignment.

Example:
| Paper ID | Publishable | Conference | Rationale                              |
|----------|-------------|------------|----------------------------------------|
| 101      | 1           | CVPR       | This paper aligns with computer vision techniques. |
| 102      | 0           | NA         | NA                                     |

---

## Extending the System
### Add New Conferences
Update the `conference_mapping` dictionary in `utils/pathway_utils.py` with new conference codes and names:
```python
conference_mapping = {
    "R006": "CVPR",
    "R007": "CVPR",
    "NEW01": "NEW_CONFERENCE"
}
```

### Change Embedding Models
Modify the `EMBEDDER_MODEL` variable in `main.py`:
```python
EMBEDDER_MODEL = "your_preferred_model_name"
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Feel free to open issues or submit pull requests for improvements and bug fixes. Contributions are welcome!

---

## Support
If you encounter any issues or have questions, please contact `your_email@example.com`.

---

## Acknowledgments
Special thanks to the teams behind Pathway, LangChain, and OpenAI for providing the foundational tools used in this project.
