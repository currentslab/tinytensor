from tinytensor.postprocessing.classification import hierarchical_dedup



def test_dedup():
    data = {
        'Computers & Electronics': 0.52014214, 
        'Computers & Electronics/Computer Hardware': 0.7732306, 
        'Computers & Electronics/Consumer Electronics': 0.8308393, 
        'Computers & Electronics/Software/Operating Systems': 0.94212234}
    output = hierarchical_dedup(data)
    assert len(output) == 3