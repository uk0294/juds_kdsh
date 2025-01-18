from langchain.llms import OpenAI

def generate_prompt_with_rationale(labeled_data, unlabelled_embedding, conference_info):
    """Generate an LLM prompt with rationale."""
    labeled_papers_prompt = "\n".join(
        f"Embedding: {data['embedding']}\nConference: {data['conference']}"
        for data in labeled_data
    )
    unlabeled_paper_prompt = f"Embedding: {unlabelled_embedding}"
    return f"""
    The labeled research papers with embeddings and conference information:
    {labeled_papers_prompt}

    The unlabelled research paper embedding:
    {unlabeled_paper_prompt}

    Conference descriptions:
    {conference_info}

    Instructions:
    1. Determine if the unlabelled paper embedding is similar to any labeled paper embedding.
    2. If yes, assign the conference of the most similar labeled paper.
    3. If no, classify the unlabelled paper into one of the conferences using the conference descriptions.
    4. Provide a rationale explaining why the unlabelled research paper was assigned to the selected conference.
    Output the assigned conference and the rationale.
    """

def call_llm_with_rationale(llm, prompt):
    """Call the LLM and extract conference and rationale."""
    response = llm(prompt)
    response_lines = response.strip().split("\n")
    assigned_conference = response_lines[0].strip()
    rationale = "\n".join(response_lines[1:]).strip()
    return assigned_conference, rationale
