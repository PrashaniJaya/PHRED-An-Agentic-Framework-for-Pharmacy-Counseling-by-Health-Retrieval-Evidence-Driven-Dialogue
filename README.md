# PHRED-An-Agentic-Framework-for-Pharmacy-Counseling-by-Health-Retrieval-Evidence-Driven-Dialogue

Medication counseling requires not only accurate information delivery but also contextual awareness, safety
validation, and clear explanation of risks capabilities that remain challenging for current conversational AI
systems. Although large language models (LLMs) have made indisputable progress in language generation and
understanding, they remain prone to hallucinations, opaque reasoning and safety risks in clinical settings.
To address these limitations, we present PHRED (Pharmacy-counseling by Health Retrieval & Evidence-
Driven Dialogue), a modular, goal-driven multi-agent framework that improves the accuracy, interpretability,
and safety of AI-assisted medication counseling. PHRED integrates Retrieval-Augmented Generation (RAG) for
evidence grounding, Knowledge Graph (KG) reasoning for structured inference, and rule-based mechanisms
for clinical safety validation. The framework operates through four stages: query decomposition, goal-oriented
decision-making, response synthesis, and session validation. User queries are decomposed into sub-questions and
classified into clinical intents, after which an orchestration layer coordinates specialized agents (RAG, KG, and
rule-based) to retrieve and validate information. A Hierarchical Task Analysis (HTA)-based follow-up module
further ensures conversational completeness by generating targeted prompts for follow-up.
As an initial pre-test case, the framework is applied to a sensitive and regulated medication counseling
scenario, demonstrating preliminary results that support its potential for safe and transparent communication.
Our framework achieved a semantic similarity of 0.89 by employing GPT-4.0 as the LLM assembler, outperforming
MedDialog GPT answers. Furthermore, it attained a faithfulness score of 0.90 utilizing Mistral as the LLM
assembler and a context precision of 0.89 employing GPT-4.0 as the LLM assembler. Overall, PHRED promotes a
scalable agentic pipeline allowing additional agents to handle specialized tasks when required. It also promotes
an interpretable architecture that enables traceability of the provided answers by combining symbolic reasoning,
structured knowledge, and RAG within a transparent, human-in-the-loop framework.

## Framework Design
This section introduces a collaborative, multi-layered framework for medication counseling. The
design of this framework followed the Design Science Research Methodology (DSRM) [19], which
includes six stages. The six stages are problem identification, defining objectives for the solution, design
and development, demonstration, evaluation and communication which is followed throughout this
research and its continuity. The problem identification and objectives for the solution are reflected
within the design of the framework. As illustrated in Figure 1, the pipeline comprises four sequential
stages: (i) Problem Identification, which decomposes complex user queries into sub-questions; (ii) Goal-
Oriented Decision, involving goal classification and agent selection for targeted information retrieval;
(iii) Response Synthesis, which integrates retrieved data, rules, and entities to generate context-aware
outputs; and (iv) Session Validation, which ensures conversational completeness by assessing the need
for further clarification. Each stage comprises one or more agents that operate collaboratively in a
structured and interoperable manner. This framework combines large language models (LLMs) with explicit
rule-based reasoning and structured knowledge representations. This hybrid design supports coherent,
goal-driven interactions while enhancing the explainability through a transparent decision logic and
traceable information flow.

<img width="1936" height="952" alt="AgenticFramework" src="https://github.com/user-attachments/assets/2a3f1745-5f15-4aa7-8236-e3e8a4fb27ae" />
