from confluent_kafka import Producer

import hopsworks
import json
import os
import logging


class Predict(object):

    def __init__(self):
        os.environ["HOPSWORKS_PUBLIC_HOST"] = "hopsworks0.logicalclocks.com"
        project = hopsworks.login()

        kafka_api = project.get_kafka_api()
        producer_config = kafka_api.get_default_config()
        producer_config['ssl.endpoint.identification.algorithm'] = 'none'
        self.producer = Producer(producer_config)

        # todo we should use decision_engine_api here, but it needs backend first
        import yaml
        dataset_api = project.get_dataset_api()
        downloaded_file_path = dataset_api.download("Resources/test_config.yml")
        with open(downloaded_file_path, "r") as f:
            config = yaml.safe_load(f)

        self.kafka_topic = '_'.join([project.name, config['name'], "logObservations"])

    def produce_to_kafka(self, data):
        try:
            # Serialize data as JSON before sending to Kafka
            message_value = json.dumps(data)

            # Produce the message to Kafka topic
            self.producer.produce(self.kafka_topic, value=message_value)

            # Ensure the message is sent to Kafka (flush)
            self.producer.flush()

            logging.info("Data forwarded to Kafka successfully")
        except Exception as e:
            logging.info(f"Error producing to Kafka: {str(e)}")

    def predict(self, inputs):
        try:
            logging.info(inputs)
            # Receive data from customer
            data = inputs[0]

            # Forward data to Kafka
            self.produce_to_kafka(data)

            return {'status': 'success'}
        except Exception as e:
            print(f"Error processing observations: {str(e)}")
            return {'status': 'error', 'message': 'Internal Server Error'}