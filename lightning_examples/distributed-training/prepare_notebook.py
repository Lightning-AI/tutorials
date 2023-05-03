# patch a few functions that would throw an error when running in the notebook
# this is because most multi-GPU plugins are only supported in script-mode
# the user is advised to read the instructions how to run the code samples given in the notebooks

from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

AcceleratorConnector.check_interactive_compatibility = lambda x: True
