from openai import OpenAI
import os
from typing import List, Dict

class StoryGenerator:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key) if api_key else None
    
    def generate_short_summary(self, document_text: str) -> str:
        if not self.client:
            return "Error: OpenAI API key is not configured."
        
        if not document_text or not document_text.strip():
            return "No content to summarize."
            
        system_prompt = "You are an expert summarizer. Your task is to provide a very concise, 2-sentence summary of the provided text. Capture the main idea effectively."
        user_prompt = f"Please summarize the following text in exactly two sentences:\n\n---\n\n{document_text}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred during summary generation: {str(e)}"
    
    def prepare_prompts(self, selected_points: List[Dict]) -> tuple[str, str]:
        if len(selected_points) == 1:
            point = selected_points[0]
            title = point.get('title', 'N/A')
            description = point.get('description', '')
            
            system_prompt = "You are a helpful assistant. Your task is to provide a concise and clear summary of the provided document text. Focus on the main points and key information."
            user_prompt = f"Please summarize the following document:\n\nTitle: {title}\n\nText:\n{description}"
            
            return system_prompt, user_prompt
        else:
            descriptions = []
            for i, point in enumerate(selected_points[:5], 1):
                desc = f"Document {i}:\nTitle: {point.get('title', 'N/A')}\nTopic: {point.get('topic_name', 'N/A')}"
                if 'description' in point and point['description']:
                    desc += f"\nDescription: {point['description']}"
                descriptions.append(desc)
            
            combined_descriptions = "\n\n---\n\n".join(descriptions)
            
            system_prompt = "You are a helpful assistant. Your task is to read several document descriptions and weave them into a single, coherent narrative that highlights the connections and common themes between them. Focus on creating a linked story."
            user_prompt = f"Here are the descriptions of several documents:\n\n{combined_descriptions}\n\nPlease create a short, engaging story that links these documents, focusing on their shared topics or narrative threads."
            
            return system_prompt, user_prompt
    
    def generate_content(self, selected_points: List[Dict]) -> str:
        if not self.client:
            return "Error: OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
        
        if not selected_points:
            return "No content to generate."
            
        system_prompt, user_prompt = self.prepare_prompts(selected_points)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while calling the OpenAI API: {str(e)}"