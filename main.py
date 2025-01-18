from utils.pathway_utils import setup_vector_store
from utils.llm_utils import generate_prompt_with_rationale, call_llm_with_rationale
from utils.io_utils import save_results_to_csv
from langchain.llms import OpenAI
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

def main():
    PATHWAY_PORT = 8765
    PATHWAY_PORT_UNLABELLED = 8766
    EMBEDDER_MODEL = "BAAI/bge-large-en-v1.5"
    SERVICE_USER_CREDENTIALS = "credentials.json"
    OUTPUT_FILE = "data/results.csv"

    # Initialize LLM
    llm = OpenAI(model="gpt-4", temperature=0)

    # Set up vector stores
    splitter = TokenCountSplitter(min_tokens=300, max_tokens=500)
    embedder = SentenceTransformerEmbedder(model=EMBEDDER_MODEL)

    client_labeled = setup_vector_store(
        object_id="1RKyDkAyW7cf09THED7Ms4LNBdeTDWnlK",
        service_user_credentials=SERVICE_USER_CREDENTIALS,
        port=PATHWAY_PORT,
        embedder=embedder,
        splitter=splitter,
    )

    client_unlabeled = setup_vector_store(
        object_id="1Y2Y0EsMalo26KcJiPYcAXh6UzgMNjh4u",
        service_user_credentials=SERVICE_USER_CREDENTIALS,
        port=PATHWAY_PORT_UNLABELLED,
        embedder=embedder,
        splitter=splitter,
    )

    # Fetch labeled data
    labeled_results = client_labeled(query="Fetch all labeled papers")
    labeled_data = [
        {"embedding": doc["embedding"], "conference": doc["metadata"]["conference"]}
        for doc in labeled_results
    ]

    # Process unlabeled papers
    conference_info = """
      prompt += '''- CVPR (Conference on Computer Vision and Pattern Recognition): Papers in the main technical program must describe high-quality, original research. Topics of interest cover all aspects of computer vision and pattern recognition including, but not limited to:
    FLIP (Flipping labels to inject poision)
    pixel deflection
    3D from multi-view and sensors  
    3D from single images  
    Adversarial attack and defense
    Autonomous driving
    Biometrics  
    Computational imaging  
    Computer vision for social good  
    Computer vision theory  
    Datasets and evaluation  
    Deep learning architectures and techniques  
    Document analysis and understanding  
    Efficient and scalable vision  
    Embodied vision: Active agents, simulation
    Event-based cameras
    Explainable computer vision  
    Humans: Face, body, pose, gesture, movement
    Image and video synthesis and generation  
    Low-level vision  
    Machine learning (other than deep learning)
    Medical and biological vision, cell microscopy
    Multimodal learning
    Optimization methods (other than deep learning)
    Photogrammetry and remote sensing  
    Physics-based vision and shape-from-X  
    Recognition: Categorization, detection, retrieval  
    Representation learning  
    Computer Vision for Robotics  
    Scene analysis and understanding  
    Segmentation, grouping and shape analysis  
    Self-, semi-, meta- and unsupervised learning
    Transfer/ low-shot/ continual/ long-tail learning  
    Transparency, fairness, accountability, privacy and ethics in vision  
    Video: Action and event understanding 
    Video: Low-level analysis, motion, and tracking  
    Vision + graphics  
    Vision, language, and reasoning  
    Vision applications and systems\n'''

    prompt += '''- EMNLP (Empirical Methods in Natural Language Processing): EMNLP 2024 aims to have a broad technical program. Relevant topics for the conference include, but are not limited to:

    Computational Social Science and Cultural Analytics
    Dialogue and Interactive Systems
    Discourse and Pragmatics
    Low-resource Methods for NLP
    Ethics, Bias, and Fairness
    Generation
    Information Extraction
    Information Retrieval and Text Mining
    Interpretability and Analysis of Models for NLP
    Linguistic theories, Cognitive Modeling and Psycholinguistics
    Machine Learning for NLP
    Machine Translation
    Multilinguality and Language Diversity
    Multimodality and Language Grounding to Vision, Robotics and Beyond
    Phonology, Morphology and Word Segmentation
    Question Answering
    Resources and Evaluation
    Semantics: Lexical, Sentence-level Semantics, Textual Inference and Other areas
    Sentiment Analysis, Stylistic Analysis, and Argument Mining
    Speech processing and spoken language understanding
    Summarization
    Syntax: Tagging, Chunking and Parsing
    NLP Applications
    Special Theme: Efficiency in Model Algorithms, Training, and Inference\n'''

    prompt += '''- KDD (Knowledge Discovery and Data Mining): For the research track, we invite submission of papers describing innovative research on all aspects of knowledge discovery and data science, ranging from theoretical foundations to novel models and algorithms for data science problems in science, business, medicine, and engineering. Visionary papers on new and emerging topics are also welcome, as are application-oriented papers that make innovative technical contributions to research. Topics of interest include, but are not limited to:

    Data Science: Methods for analyzing social networks, time series, sequences, streams, text, web, graphs, rules, patterns, logs, IoT data, spatio-temporal data, biological data, scientific and business data; recommender systems, computational advertising, multimedia, finance, bioinformatics.
    Big Data: Large-scale systems for data analysis, machine learning, optimization, sampling, summarization; parallel and distributed data science (cloud, map-reduce, federated learning); novel algorithmic and statistical techniques for big data; algorithmically-efficient data transformation and integration.
    Foundations: Models and algorithms, asymptotic analysis; model selection, dimensionality reduction, relational/structured learning, matrix and tensor methods, probabilistic and statistical methods; deep learning, transfer learning, representation learning, meta learning, reinforcement learning; classification, clustering, regression, semi-supervised learning, self-supervised learning, few shot learning and unsupervised learning; personalization, security and privacy, visualization; fairness, interpretability, ethics and robustness.
    it includes debruijn graphs 
    for example the abstract of research paper in KDD-Parkinson’s disease (PD) is a progressive neurodegenerative disorder that leads to motor symptoms, including gait
impairment. The effectiveness of levodopa therapy, a common treatment for PD, can fluctuate, causing periods of
improved mobility ("on" state) and periods where symptoms re-emerge ("off" state). These fluctuations impact
gait speed and increase in severity as the disease progresses. This paper proposes a transformer-based method that
uses both Received Signal Strength Indicator (RSSI) and accelerometer data from wearable devices to enhance
indoor localization accuracy. A secondary goal is to determine if indoor localization, particularly in-home gait
speed features (like the time to walk between rooms), can be used to identify motor fluctuations by detecting if a
person with PD is taking their levodopa medication or not. The method is evaluated using a real-world dataset
collected in a free-living setting, where movements are varied and unstructured. Twenty-four participants, living
in pairs (one with PD and one control), resided in a sensor-equipped smart home for five days. The results show
that the proposed network surpasses other methods for indoor localization. The evaluation of the secondary goal
reveals that accurate room-level localization, when converted into in-home gait speed features, can accurately
predict whether a PD participant is taking their medication or not.\n'''

    prompt += '''- NeurIPS (Neural Information Processing Systems): This conference invites submissions presenting new and original research on a variety of machine learning topics.•	General Machine Learning: Including classification, unsupervised learning, transfer learning, and reinforcement learning.
	•	Neuroscience and Cognitive Science: Studies that bridge machine learning with insights from neuroscience and cognitive science.
	•	Robotics: Integration of machine learning techniques in robotic perception, control, and decision-making..
	•	Theory: Theoretical foundations underpinning machine learning algorithms and statistical learning.
	•	Applications: Innovative applications of machine learning in various domains, finance, and social sciences.
    
    for example abstract of research paper 
    
    eg. 1 Regression tasks, while aiming to model relationships across the entire input space,

are often constrained by limited training data. Nevertheless, if the hypothesis func-
tions can be represented effectively by the data, there is potential for identifying a

model that generalizes well. This paper introduces the Neural Restricted Isometry
Property (NeuRIPs), which acts as a uniform concentration event that ensures all
shallow ReLU networks are sketched with comparable quality. To determine the
sample complexity necessary to achieve NeuRIPs, we bound the covering numbers

of the networks using the Sub-Gaussian metric and apply chaining techniques. As-
suming the NeuRIPs event, we then provide bounds on the expected risk, applicable

to networks within any sublevel set of the empirical risk. Our results show that all
networks with sufficiently small empirical risk achieve uniform generalization.\n'''

    prompt += '''- TMLR (Transactions on Machine Learning Research): TMLR’s objective is to publish original papers that contribute to the understanding of the computational and mathematical principles that enable intelligence through learning, be it in brains or in machines.
It includes papers with positional encoding 
To this end, TMLR invites authors to submit papers that contain
it includes topics TFGNN , Training Free Graph Neural Network 
new algorithms with sound empirical validation, optionally with justification of theoretical, psychological, or biological nature;
experimental and/or theoretical studies yielding new insight into the design and behavior of learning in intelligent systems;
accounts of applications of existing techniques that shed light on the strengths and weaknesses of the methods;
formalization of new learning tasks (e.g., in the context of new applications) and of methods for assessing performance on those tasks;
development of new analytical frameworks that advance theoretical studies of practical learning methods;
computational models of natural learning systems at the behavioral or neural level;
reproducibility studies of previously published results or claims;
new approaches for analysis, visualization, and understanding of artificial or biological learning systems;
surveys that draw new connections, highlight trends, and suggest new problems in an area.


for example the abstract of TMLR research papers 

e.g 1 This research examines a specific category of structured nonconvex-nonconcave min-max problems that demon-
strate a characteristic known as weak Minty solutions. This concept, which has only recently been defined, has

already demonstrated its effectiveness by encompassing various generalizations of monotonicity at the same time.
We establish new convergence findings for an enhanced variant of the optimistic gradient method (OGDA) within
this framework, achieving a convergence rate of 1/k for the most effective iteration, measured by the squared
operator norm, a result that aligns with the extragradient method (EG). Furthermore, we introduce a modified
version of EG that incorporates an adaptive step size, eliminating the need for prior knowledge of the problem’s
specific parameters.

e.g 2 Deep generative models, particularly diffusion models, are a significant family within deep learning. This study
provides a precise upper limit for the Wasserstein distance between a learned distribution by a diffusion model
and the target distribution. In contrast to earlier research, this analysis does not rely on presumptions regarding
the learned score function. Furthermore, the findings are applicable to any data-generating distributions within
restricted instance spaces, even those lacking a density relative to the Lebesgue measure, and the upper limit is not
exponentially dependent on the ambient space dimension. The primary finding expands upon recent research by
Mbacke et al. (2023), and the proofs presented are fundamental.\n'''
    """
    unlabelled_results = client_unlabeled(query="Fetch all unlabelled papers")
    results = []

    for paper in unlabelled_results:
        unlabelled_embedding = paper["embedding"]
        prompt = generate_prompt_with_rationale(labeled_data, unlabelled_embedding, conference_info)
        assigned_conference, rationale = call_llm_with_rationale(llm, prompt)
        results.append({
            "Paper ID": paper["id"],
            "Classification": f"Conference: {assigned_conference}",
            "Reasoning": rationale
        })

    # Save results
    save_results_to_csv(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
