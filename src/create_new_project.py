import logging
from labelbox import Client, Dataset, Project, OntologyBuilder, MediaType, Ontology
from dotenv import load_dotenv
import os

# Configure logging
log_file = 'labelbox_manager.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

class LabelboxManager:
    def __init__(self, api_key):
        self.client = Client(api_key=api_key)
        self.dataset = None
        self.project = None
        self.ontology = None

    def create_dataset(self, dataset_name):
        """Create a new dataset."""
        try:
            self.dataset = self.client.create_dataset(name=dataset_name)
            logging.info(f"Dataset created with name: {self.dataset.name}, ID: {self.dataset.uid}")
            return self.dataset.uid
        except Exception as e:
            logging.error(f"Failed to create dataset: {e}")
            raise

    def delete_dataset(self, dataset_id):
        """Delete an existing dataset."""
        try:
            dataset = self.client.get_dataset(dataset_id)
            dataset.delete()
            logging.info(f"Dataset with ID {dataset_id} deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete dataset: {e}")
            raise

    def create_project(self, project_name):
        """Create a new project."""
        try:
            self.project = self.client.create_project(name=project_name, media_type=MediaType.Video)
            logging.info(f"Project created with name: {self.project.name}, ID: {self.project.uid}")
            return self.project.uid
        except Exception as e:
            logging.error(f"Failed to create project: {e}")
            raise

    def delete_project(self, project_id):
        """Delete an existing project."""
        try:
            project = self.client.get_project(project_id)
            project.delete()
            logging.info(f"Project with ID {project_id} deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete project: {e}")
            raise

    def create_ontology(self, project_id, ontology_name, classifications):
        """Create a new ontology and attach it to the project."""
        try:
            ontology_builder = OntologyBuilder(classifications=classifications)
            normalized_ontology = ontology_builder.build()  # Build the ontology schema
            serialized_ontology = normalized_ontology.asdict()  # Serialize it

            self.ontology = self.client.create_ontology(name=ontology_name, normalized=serialized_ontology)
            logging.info(f"Ontology created with name: {self.ontology.name}, ID: {self.ontology.uid}")

            self.project = self.client.get_project(project_id)
            self.project.setup_editor(self.ontology)
            logging.info(f"Ontology attached to project with ID: {self.project.uid}")
            return self.ontology.uid
        except Exception as e:
            logging.error(f"Failed to create ontology: {e}")
            raise

# Usage Example
load_dotenv()
api_key = os.getenv('LABELBOX_API_KEY')
manager = LabelboxManager(api_key)

# Define classifications (video labels)
classifications = [
    {
        "type": "radio",
        "instructions": "Select the action",
        "options": [
            {"label": "Likken"},
            {"label": "Elkaar likken"},
            {"label": "Eten"},
            {"label": "Drinken"},
            {"label": "Staan"},
            {"label": "Herkauwen (staand)"},
            {"label": "Liggen (rusten)"},
            {"label": "Liggen (slapen)"},
            {"label": "Liggen (herkauwen)"},
            {"label": "Kopstoot"},
            {"label": "Bewegen"},
            {"label": "Ontlasten (defeceren)"},
            {"label": "Ontlasten (urineren)"},
            {"label": "Slapen"},
            {"label": "Rusten"},
            {"label": "Grazen"},
            {"label": "Gras afbijten"},
            {"label": "Gras kauwen en doorslikken"},
            {"label": "Loeien"},
            {"label": "Hoofd omhoog"},
            {"label": "Hoofd omlaag"},
            {"label": "Opgegeven hoofd"},
            {"label": "Hoofd op flank"},
            {"label": "Hangende staart"},
            {"label": "Staart in beweging"},
            {"label": "Onbekend (liggen)"},
            {"label": "Onbekend (staan)"}
        ]
    }
]

project_id = 'clxonpwjg080g07288mf33l3q'

# Create a new ontology and attach to project
ontology_name = "SingleCow Video Actions Ontology"
ontology_id = manager.create_ontology(project_id, ontology_name, classifications)

print(f"Ontology ID: {ontology_id}")
