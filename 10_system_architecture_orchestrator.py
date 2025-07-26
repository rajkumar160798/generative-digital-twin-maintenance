import argparse
import logging
import yaml

# Assume all required modules exist within the src package
try:
    from src.data.xjtu import load_xjtu_data
    from src.data.ai4i import load_ai4i_data
    from src.data.cmapss import load_cmapss_data
    from src.models.gan import generate_faults
    from src.explainability.shap import run_shap_analysis
    from src.models.lstm import predict_rul as predict_rul_lstm
    from src.models.prophet_rul import predict_rul as predict_rul_prophet
    from src.models.rl_scheduler import schedule_maintenance as schedule_maintenance_rl
    from src.report.report_generator import create_report
except Exception:  # pragma: no cover - modules may not exist during static analysis
    pass


class DigitalTwinSystem:
    """Orchestrates the generative digital twin pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.fault_data = None
        self.explanations = None
        self.rul_predictions = None
        self.maintenance_schedule = None
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------ Data Loading ------------------
    def load_data(self):
        """Load dataset based on configuration."""
        dataset = self.config.get("dataset", "xjtu").lower()
        self.logger.info("Loading data for dataset: %s", dataset)
        if dataset == "xjtu":
            self.data = load_xjtu_data()
        elif dataset == "ai4i":
            self.data = load_ai4i_data()
        elif dataset == "cmapss":
            self.data = load_cmapss_data()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return self.data

    # ------------------ Fault Simulation ------------------
    def simulate_faults(self):
        """Simulate faults using a GAN model."""
        threshold = self.config.get("fault_threshold", 0.8)
        self.logger.info("Simulating faults with threshold: %s", threshold)
        self.fault_data = generate_faults(self.data, threshold=threshold)
        return self.fault_data

    # ------------------ Explainability ------------------
    def explain_faults(self):
        """Run SHAP-based analysis and attribution drift."""
        self.logger.info("Running SHAP explainability analysis")
        self.explanations = run_shap_analysis(self.fault_data)
        return self.explanations

    # ------------------ RUL Prediction ------------------
    def predict_rul(self):
        """Predict Remaining Useful Life using chosen model."""
        model = self.config.get("rul_model", "lstm").lower()
        self.logger.info("Predicting RUL using model: %s", model)
        if model == "lstm":
            self.rul_predictions = predict_rul_lstm(self.fault_data)
        elif model == "prophet":
            self.rul_predictions = predict_rul_prophet(self.fault_data)
        else:
            raise ValueError(f"Unsupported RUL prediction model: {model}")
        return self.rul_predictions

    # ------------------ Maintenance Scheduling ------------------
    def schedule_maintenance(self):
        """Optimize maintenance schedule using RL."""
        self.logger.info("Scheduling maintenance with RL policy")
        self.maintenance_schedule = schedule_maintenance_rl(self.rul_predictions)
        return self.maintenance_schedule

    # ------------------ Reporting ------------------
    def generate_report(self):
        """Generate a summary report for the pipeline."""
        if not self.config.get("generate_report", True):
            self.logger.info("Report generation disabled")
            return None
        self.logger.info("Generating report")
        report_path = create_report(
            data=self.data,
            faults=self.fault_data,
            explanations=self.explanations,
            rul=self.rul_predictions,
            schedule=self.maintenance_schedule,
        )
        return report_path

    # ------------------ Visualization ------------------
    def visualize_pipeline(self, output_path: str = "pipeline_flowchart.png"):
        """Visualize the pipeline steps as a flowchart."""
        self.logger.info("Creating pipeline visualization")
        try:
            from graphviz import Digraph
        except ImportError:
            self.logger.error("graphviz is required for visualization. Please install it.")
            return None

        g = Digraph(format="png")
        g.attr(rankdir="LR")

        g.node("A", "Load Data")
        g.node("B", "Simulate Faults")
        g.node("C", "Explain Faults")
        g.node("D", "Predict RUL")
        g.node("E", "Schedule Maintenance")
        g.node("F", "Generate Report")

        g.edges(["AB", "BC", "CD", "DE", "EF"])

        g.render(output_path, cleanup=True)
        self.logger.info("Pipeline visualization saved to %s", output_path)
        return output_path


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    config = load_config(config_path)
    system = DigitalTwinSystem(config)

    system.load_data()
    system.simulate_faults()
    system.explain_faults()
    system.predict_rul()
    system.schedule_maintenance()
    system.generate_report()
    system.visualize_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Twin System Orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
