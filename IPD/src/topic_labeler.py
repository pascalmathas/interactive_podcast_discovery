from bertopic.representation._base import BaseRepresentation

class GPT3TopicLabeler(BaseRepresentation):
    def __init__(self, system_prompt, prompt_template):
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
    
    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        return {}