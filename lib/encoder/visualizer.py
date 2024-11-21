import os

from .IdentityBert import IdentityBert
from transformers import pipeline
from spacy import displacy
import spacy
from typing import Tuple, Set, Dict, List, Optional
import torch
from ..utilities import Timer
from spacy.tokens import Doc, Span
from transformers import pipeline
from transformers.pipelines import Pipeline
from spacy.language import Language
import html  # For escaping HTML entities
import re
from ..utilities import Timer, Logger

class Visualizer(IdentityBert):
    _CallerPrefix = "Visualizer"

    _entity_colors: Dict[str, str] = {
        "identity": "#DFF4E1",    # Lighter Medium Slate Blue for predicted organizations
        "action": "#F4D1C1",      # Lighter Indian Red for predicted actions
        "relation": "#DFF4E1",    # Lighter Lime Green for predicted relations
    }

    def __init__(self, identity_checkpoint_path: str):
        """
        Initializes the Visualizer with a specified identity checkpoint.

        Args:
            identity_checkpoint_path (str): Path to the identity model checkpoint.
        """
        super().__init__(identity_checkpoint_path)  # Initialize the superclass
        # Remove the existing classifier_pipeline if it exists
        if hasattr(self, 'classifier_pipeline'):
            del self.classifier_pipeline
        # Initialize a new NER pipeline without aggregation
        self.classifier_pipeline: Pipeline = pipeline(
            "ner",
            model=identity_checkpoint_path,
            device=self.device,  # Ensure 'self.device' is defined in IdentityBert
            aggregation_strategy="first"  # No aggregation
        )

    @classmethod
    def get_entity_colors(cls) -> Dict[str, str]:
        """
        Retrieves the current entity colors.

        Returns:
            Dict[str, str]: A dictionary mapping entity types to their colors.
        """
        return cls._entity_colors

    @classmethod
    def set_entity_colors(cls, entity_colors: Dict[str, str]) -> None:
        """
        Sets the entity colors.

        Args:
            entity_colors (Dict[str, str]): A dictionary mapping entity types to colors.
        """
        cls._entity_colors = entity_colors

    @staticmethod
    def sanitize_metadata(metadata: str) -> str:
        """
        Escapes HTML entities and preserves whitespace in metadata.

        Args:
            metadata (str): The raw metadata string.

        Returns:
            str: Sanitized metadata string safe for HTML rendering.
        """
        # Escape HTML entities
        escaped_metadata = html.escape(metadata)
        # Optionally, replace tabs with spaces or another representation
        escaped_metadata = escaped_metadata.replace('\t', '&emsp;')  # Replaces tab with em space
        # Newlines will be handled by CSS 'white-space: pre-wrap;'
        return escaped_metadata

    @staticmethod
    def ner_create_spacy_doc(raw_text: str, entities: List[Dict], nlp: Language) -> Doc:
        """
        Creates a spaCy Doc object with the specified entities.

        Args:
            raw_text (str): The original text.
            entities (List[Dict]): A list of entities with 'start', 'end', and 'entity_group'.
            nlp (Language): A spaCy language model.

        Returns:
            Doc: A spaCy Doc object with entities.
        """
        doc = nlp.make_doc(raw_text)
        ents: List[Span] = []
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'], label=ent['entity_group'])
            if span is not None:
                ents.append(span)
            else:
                # Optionally handle cases where char_span returns None
                print(f"Warning: Could not create span for entity {ent}")
        doc.ents = ents
        return doc

    @staticmethod
    def visualize_predictions(pred_doc: Doc, metadata: str = "", options: Optional[Dict] = None) -> str:
        """
        Renders the visualization of predictions using spaCy's displacy.

        Args:
            pred_doc (Doc): The spaCy Doc object with predicted entities.
            metadata (str, optional): Additional metadata to display. Defaults to "".
            options (Optional[Dict], optional): Visualization options. Defaults to None.

        Returns:
            str: Rendered HTML string.
        """
        # Generate the displacy HTML with page=False to get the snippet only
        pred_html = displacy.render(pred_doc, style="ent", page=False, options=options)
        if len(metadata):
            sanitized_metadata = Visualizer.sanitize_metadata(metadata)
        else:
            sanitized_metadata = ""
        # Add custom CSS for nested labels
        custom_css = """
        <style>
        .entity {
            display: inline-block;
            position: relative;
            padding: 2px 4px;
            margin: 0 2px;
            line-height: 1.5;
            border-radius: 4px;
        }
        .entity .label {
            position: absolute;
            bottom: -1.5em;
            left: 0;
            background: white;
            color: black;
            font-size: 0.8em;
            padding: 1px 4px;
            border-radius: 2px;
            border: 1px solid #ccc;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            white-space: nowrap;
        }
        </style>
        """

        # Use Flexbox for layout to prevent overlapping
        if len(sanitized_metadata):
            rendered_html = f"""
             {custom_css}
                <h2>NER Entity Predictions</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1 1 60%; min-width: 300px; border: 1px solid black; padding: 10px; box-sizing: border-box;">
                        <h3>Predicted Entities</h3>
                        {pred_html}
                    </div>
                    <div style="
                        flex: 1 1 30%; 
                        min-width: 200px; 
                        border: 1px solid black; 
                        padding: 10px; 
                        box-sizing: border-box; 
                        background-color: #f9f9f9; 
                        white-space: pre-wrap;  /* Preserves whitespace */
                        max-height: 400px;      /* Optional: set a max height */
                        overflow: auto;         /* Adds scrollbar if content exceeds max-height */
                    ">
                        <strong>Metadata:</strong><br>
                        {sanitized_metadata}
                    </div>
                </div>
            """
        else:
            rendered_html = f"""
             {custom_css}
                <h2>NER Entity Predictions</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1 1 100%; min-width: 300px; border: 1px solid black; padding: 10px; box-sizing: border-box;">
                        <h3>Predicted Entities</h3>
                        {pred_html}
                    </div>
                </div>
            """
        return rendered_html

    @staticmethod
    def ner_clean_predictions(predictions: List[Dict], text: str) -> List[Dict]:
        """
        Cleans the raw NER predictions by ensuring proper entity spans.

        Since aggregation_strategy='first' is used, entities are already aggregated.
        This function can be simplified or removed if no further cleaning is needed.
        """
        # With aggregation_strategy='first', entities are already aggregated.
        # Therefore, we can return the predictions as is, possibly ensuring they are unique.
        unique_entities = []
        seen = set()
        for ent in predictions:
            key = (ent['start'], ent['end'], ent['entity_group'])
            if key not in seen:
                seen.add(key)
                unique_entities.append({
                    "start": ent['start'],
                    "end": ent['end'],
                    "entity_group": ent['entity_group']
                })
        return unique_entities

    @torch.inference_mode()
    def __call__(self, raw_text: str, metadata: str = "") -> str:
        """
        Processes the raw text to identify entities and visualize them.

        Args:
            raw_text (str): The input text to process.
            metadata (str, optional): Additional metadata to display. Defaults to "".

        Returns:
            str: Rendered HTML visualization of the entities.
        """
        raw_text = self.remove_urls(raw_text)
        raw_text = self._tokenize(raw_text)
        with Timer():
            # Obtain NER predictions
            output = self.classifier_pipeline(raw_text)

        # Temporary lists to store entities with their confidence scores
        temp_identities: List[Tuple[str, float]] = []
        temp_relation: List[Tuple[str, float]] = []
        seen_identities = set()

        for ent in output:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            ent_score = ent.get('score', 0.0)  # Get the confidence score
            if ent_label == 'identity':
                if ent_text not in seen_identities:
                    temp_identities.append((ent_text, ent_score))
                    seen_identities.add(ent_text)
            elif ent_label == 'relation':
                temp_relation.append(ent_text)

        temp_identities.sort(key=lambda x: x[1], reverse=True)

        # Logging: Convert entities with scores to strings for better readability
        identities_str = '\n'.join(
            [f"{entity}, confidence = {score:.2f}"
             for entity, score in temp_identities])
        relation_str = f"{temp_relation}"

        metadata += '\nReported identities: ' + '\n' + identities_str + '\nReported relation: ' + '\n' + relation_str

        # Clean the predictions
        cleaned_output = self.ner_clean_predictions(output, raw_text)

        # Initialize a blank spaCy model
        nlp = spacy.blank("en")

        # Create a spaCy Doc with the predicted entities
        pred_doc = self.ner_create_spacy_doc(raw_text, cleaned_output, nlp)

        # Render the visualization using the current entity colors
        html = self.visualize_predictions(
            pred_doc,
            metadata=metadata,
            options={"colors": self.get_entity_colors()}
        )

        return html
