from .IdentityBert import IdentityBert
from transformers import pipeline
from spacy import displacy
import spacy
from typing import Tuple, Set, Dict, List
import torch
from ..utilities import Timer

class Visualizer(IdentityBert):
    _entity_colors = {
        "identity": "#9AC0F4",  # Lighter Medium Slate Blue for predicted organizations
        "action": "#F5B8B8",  # Lighter Indian Red for predicted actions
        "relation": "#7CFC7C",  # Lighter Lime Green for predicted relations
    }

    def __init__(self, identity_checkpoint_path: str):
        super().__init__(identity_checkpoint_path)  # Call the superclass constructor
        del self.classifier_pipeline
        self.classifier_pipeline = pipeline("ner",
                                            model=identity_checkpoint_path,
                                            device=self.device,
                                            aggregation_strategy="none") # without aggregation strategy

    @classmethod
    def get_entity_colors(cls):
        return cls._entity_colors

    @classmethod
    def set_entity_colors(cls, entity_colors: Dict):
        cls._entity_colors = entity_colors

    @staticmethod
    def ner_create_spacy_doc(raw_text: str, entities: List[Dict], nlp: spacy):
        doc = nlp.make_doc(raw_text)
        ents = []
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'], label=ent['entity_group'])
            if span is not None:
                ents.append(span)
        doc.ents = ents
        return doc

    @staticmethod
    def visualize_predictions(pred_doc, metadata="", options=None):
        pred_html = displacy.render(pred_doc, style="ent", page=True, options=options)
        rendered_html = \
            f"""
                <h2>NER Entity Predictions</h2>
                <div style="display: flex; justify-content: space-around; position: relative;">
                    <div style="width: 100%; border: 1px solid black;">
                        <h3>Predicted</h3>
                        {pred_html}
                    </div>
                    <div style="position: absolute; top: 0; right: 0; background-color: #fff; padding: 5px; border: 1px solid black;">
                        <strong>Metadata:</strong> {metadata}
                    </div>
                </div>
            """
        return rendered_html


    @staticmethod
    def ner_clean_predictions(predictions, text):

        def adjust_for_multiline(text, start, end):
            """Adjust entity start and end positions for multi-line text."""
            new_start = start
            new_end = end
            lines = text.split('\n')

            # Adjust start position
            for line in lines:
                if new_start > len(line):
                    new_start -= (len(line) + 1)  # +1 for the newline character
                else:
                    break

            # Adjust end position
            for line in lines:
                if new_end > len(line):
                    new_end -= (len(line) + 1)  # +1 for the newline character
                else:
                    break

            return new_start, new_end

        entities = []
        current_entity = None
        current_label = None

        for item in predictions:
            if item['entity'].startswith('B-') or (item['entity'].startswith('I-') and current_entity is None):
                if current_entity is not None:
                    entities.append({
                        "start": current_entity['start'],
                        "end": current_entity['end'],
                        "entity_group": current_label
                    })
                current_entity = {"start": item['start'], "end": item['end']}
                current_label = item['entity'][2:]
            elif item['entity'].startswith('I-') and current_label == item['entity'][2:]:
                current_entity['end'] = item['end']
            else:
                if current_entity is not None:
                    entities.append({
                        "start": current_entity['start'],
                        "end": current_entity['end'],
                        "entity_group": current_label
                    })
                current_entity = None
                current_label = None

        if current_entity is not None:
            entities.append({
                "start": current_entity['start'],
                "end": current_entity['end'],
                "entity_group": current_label
            })

        # Handle multi-line text
        for entity in entities:
            entity['start'], entity['end'] = adjust_for_multiline(text, entity['start'], entity['end'])

        return entities

    @torch.inference_mode()
    def __call__(self, raw_text: str, metadata: str) -> str:
        with Timer() as timer:
            output = self.classifier_pipeline(raw_text)

        # Clean the output
        cleaned_output = self.ner_clean_predictions(output, raw_text)
        nlp = spacy.blank("en")
        pred_doc = self.ner_create_spacy_doc(raw_text, cleaned_output, nlp)
        # Render the visualization
        html = self.visualize_predictions(pred_doc,
                                          metadata=metadata,
                                          options={"colors": Visualizer._entity_colors})

        return html