"""
Event Bus Publishers

Multi-publisher that fans out events to Azure Service Bus, Kafka, and MQTT.
Each publisher runs in a background thread with an internal queue.
"""

import os
import json
import queue
import threading
from typing import Dict, Optional
from datetime import datetime


class AzureServiceBusPublisher:
    """Azure Service Bus publisher."""
    
    def __init__(self, connection_string: str, topic_name: str = "trailer-events"):
        """
        Initialize Azure Service Bus publisher.
        
        Args:
            connection_string: Azure Service Bus connection string
            topic_name: Topic name for publishing
        """
        self.connection_string = connection_string
        self.topic_name = topic_name
        self.queue = queue.Queue(maxsize=1000)
        self.running = False
        self.thread = None
        
        try:
            from azure.servicebus import ServiceBusClient, ServiceBusMessage
            self.ServiceBusClient = ServiceBusClient
            self.ServiceBusMessage = ServiceBusMessage
            self.client = ServiceBusClient.from_connection_string(connection_string)
            self.sender = self.client.get_topic_sender(topic_name)
        except ImportError:
            print("Warning: azure-servicebus not installed")
            self.sender = None
    
    def start(self):
        """Start background publishing thread."""
        if self.sender is None:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background publishing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def publish(self, event: Dict):
        """
        Queue an event for publishing.
        
        Args:
            event: Event dict to publish
        """
        if self.sender is None:
            return
        
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            print("Warning: Azure Service Bus queue full, dropping event")
    
    def _publish_loop(self):
        """Background loop that publishes queued events."""
        while self.running:
            try:
                event = self.queue.get(timeout=1.0)
                message = self.ServiceBusMessage(json.dumps(event))
                self.sender.send_messages(message)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error publishing to Azure Service Bus: {e}")


class KafkaPublisher:
    """Kafka publisher."""
    
    def __init__(self, broker: str, topic: str = "trailer-events"):
        """
        Initialize Kafka publisher.
        
        Args:
            broker: Kafka broker address (e.g., "localhost:29092")
            topic: Kafka topic name
        """
        self.broker = broker
        self.topic = topic
        self.queue = queue.Queue(maxsize=1000)
        self.running = False
        self.thread = None
        
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except ImportError:
            print("Warning: kafka-python not installed")
            self.producer = None
    
    def start(self):
        """Start background publishing thread."""
        if self.producer is None:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background publishing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        if self.producer:
            self.producer.close()
    
    def publish(self, event: Dict):
        """
        Queue an event for publishing.
        
        Args:
            event: Event dict to publish
        """
        if self.producer is None:
            return
        
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            print("Warning: Kafka queue full, dropping event")
    
    def _publish_loop(self):
        """Background loop that publishes queued events."""
        while self.running:
            try:
                event = self.queue.get(timeout=1.0)
                self.producer.send(self.topic, value=event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error publishing to Kafka: {e}")


class MQTTPublisher:
    """MQTT publisher."""
    
    def __init__(self, broker: str, port: int = 1883, topic: str = "trailer/events"):
        """
        Initialize MQTT publisher.
        
        Args:
            broker: MQTT broker host
            port: MQTT broker port
            topic: MQTT topic name
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.queue = queue.Queue(maxsize=1000)
        self.running = False
        self.thread = None
        
        try:
            import paho.mqtt.client as mqtt
            self.mqtt = mqtt
            self.client = mqtt.Client()
            self.client.connect(broker, port, 60)
            self.client.loop_start()
        except ImportError:
            print("Warning: paho-mqtt not installed")
            self.client = None
    
    def start(self):
        """Start background publishing thread."""
        if self.client is None:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background publishing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish(self, event: Dict):
        """
        Queue an event for publishing.
        
        Args:
            event: Event dict to publish
        """
        if self.client is None:
            return
        
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            print("Warning: MQTT queue full, dropping event")
    
    def _publish_loop(self):
        """Background loop that publishes queued events."""
        while self.running:
            try:
                event = self.queue.get(timeout=1.0)
                payload = json.dumps(event)
                self.client.publish(self.topic, payload)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error publishing to MQTT: {e}")


class MultiPublisher:
    """
    Multi-publisher that fans out events to enabled buses.
    """
    
    def __init__(self):
        """Initialize multi-publisher."""
        self.publishers = []
        self._init_publishers()
    
    def _init_publishers(self):
        """Initialize enabled publishers based on environment variables."""
        # Azure Service Bus
        azure_conn = os.getenv('AZURE_SERVICEBUS_CONNECTION')
        if azure_conn:
            topic = os.getenv('AZURE_SERVICEBUS_TOPIC', 'trailer-events')
            pub = AzureServiceBusPublisher(azure_conn, topic)
            pub.start()
            self.publishers.append(pub)
            print(f"Enabled Azure Service Bus publisher (topic: {topic})")
        
        # Kafka
        if os.getenv('KAFKA_ENABLED', 'false').lower() == 'true':
            broker = os.getenv('KAFKA_BROKER', 'localhost:29092')
            topic = os.getenv('KAFKA_TOPIC', 'trailer-events')
            pub = KafkaPublisher(broker, topic)
            pub.start()
            self.publishers.append(pub)
            print(f"Enabled Kafka publisher (broker: {broker}, topic: {topic})")
        
        # MQTT
        if os.getenv('MQTT_ENABLED', 'false').lower() == 'true':
            broker = os.getenv('MQTT_BROKER', 'localhost')
            port = int(os.getenv('MQTT_PORT', '1883'))
            topic = os.getenv('MQTT_TOPIC', 'trailer/events')
            pub = MQTTPublisher(broker, port, topic)
            pub.start()
            self.publishers.append(pub)
            print(f"Enabled MQTT publisher (broker: {broker}:{port}, topic: {topic})")
    
    def publish(self, event: Dict):
        """
        Publish event to all enabled buses.
        
        Args:
            event: Event dict to publish
        """
        for pub in self.publishers:
            pub.publish(event)
    
    def stop(self):
        """Stop all publishers."""
        for pub in self.publishers:
            pub.stop()
    
    def get_queue_depth(self) -> int:
        """
        Get total queue depth across all publishers.
        
        Returns:
            Total number of queued events
        """
        total = 0
        for pub in self.publishers:
            if hasattr(pub, 'queue'):
                total += pub.queue.qsize()
        return total


