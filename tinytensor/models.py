import onnxruntime as ort



class Model():
    def __init__(self, configuration) -> None:
        # handle code
        self.ort = ort.InferenceSession(configuration['model'])

    def forward(self, **kwargs):
        # pass to model
        # iterative decoding output
        return self.ort.run(None, kwargs)

