

class Postprocessing():

    def forward(self, outputs, chains):
        '''
            outputs: Outputs results from inference models
            chains: Other results from previous models
        '''
        raise NotImplemented()

