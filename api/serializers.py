from rest_framework import serializers



class TrainingDataSerializer(serializers.Serializer):
    """
    Serializer to validate the training data
    """
    
    model_name = serializers.CharField(max_length=30)
    selected_column = serializers.CharField(max_length=30)
    ignore_columns = serializers.CharField(max_length=30, required=False)
    training_file = serializers.FileField()
    

    
class PredictionSerializer(serializers.Serializer):
    """
    A ModelSerializer that takes an additional `fields` argument that
    controls which fields should be displayed.
    """

    model_id = serializers.CharField(max_length=30)
    parameters = serializers.CharField(max_length=1000)
