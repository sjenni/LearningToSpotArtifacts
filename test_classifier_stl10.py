from Preprocessor import Preprocessor
from eval.SDNetTester import SDNetTester
from datasets.STL10 import STL10
from models.SpotNet import SNet

target_shape = [96, 96, 3]
model = SNet(None, batch_size=200, target_shape=target_shape, disc_pad='SAME', tag='default')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape)
tester = SDNetTester(model, data, preprocessor)
tester.test_classifier(num_conv_trained=0)
