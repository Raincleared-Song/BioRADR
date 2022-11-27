from .FineTuneModel import FineTuneModel
from .DenoiseModel import DenoiseModel
from .PretrainModel import PretrainModel
from .DocuNetFinetune import DocuNetFinetune
from .DocuNetDenoise import DocuNetDenoise
from .CoTrainModel import CoTrainModel

name_to_model = {
    'DenoiseModel': DenoiseModel,
    'FineTuneModel': FineTuneModel,
    'PretrainModel': PretrainModel,
    'DocuNetDenoise': DocuNetDenoise,
    'DocuNetFinetune': DocuNetFinetune,
    'CoTrainModel': CoTrainModel,
}
