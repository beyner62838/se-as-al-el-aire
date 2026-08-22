"""Pruebas de regresión para las rutas públicas de la aplicación."""

import json
from unittest.mock import patch

import numpy as np
from django.core.cache import cache
from django.test import Client, TestCase

from proyecto.ml.sequence_dataset import build_sequence_features


class CameraPredictTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = Client()

    def test_rechaza_solicitudes_no_post(self):
        response = self.client.get('/camera_predict/')

        self.assertEqual(response.status_code, 405)

    def test_rechaza_contenido_que_no_es_json(self):
        response = self.client.post(
            '/camera_predict/', data='imagen', content_type='text/plain'
        )

        self.assertEqual(response.status_code, 415)

    def test_rechaza_json_invalido_sin_cargar_modelos(self):
        response = self.client.post(
            '/camera_predict/',
            data='no es json',
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    def test_requiere_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        response = csrf_client.post(
            '/camera_predict/',
            data=json.dumps({'image': 'abc'}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 403)

    def test_pagina_principal_entrega_cookie_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)

        response = csrf_client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('csrftoken', response.cookies)

    @patch('proyecto.views.get_or_train_landmark_model')
    @patch('proyecto.views.get_or_train_image_model')
    @patch('proyecto.views.predict_camera_image')
    def test_responde_prediccion_valida(
        self,
        predict_camera_image,
        get_image_model,
        get_landmark_model,
    ):
        predict_camera_image.return_value = (
            'A',
            0.95,
            {
                'hand_detected': True,
                'preprocess_mode': 'mediapipe-landmarks',
                'prediction_mode': 'hybrid-landmarks-image',
            },
        )
        response = self.client.post(
            '/camera_predict/',
            data=json.dumps({'image': 'imagen-simulada'}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['raw_label'], 'A')
        get_image_model.assert_called_once()
        get_landmark_model.assert_called_once()


class DynamicSequenceTests(TestCase):
    def test_endpoint_rechaza_solicitudes_no_post(self):
        response = self.client.get('/sequence_predict/')

        self.assertEqual(response.status_code, 405)

    def test_endpoint_rechaza_secuencia_vacia(self):
        response = self.client.post(
            '/sequence_predict/',
            data=json.dumps({'frames': []}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    @patch('proyecto.views.DYNAMIC_MODEL_PATH')
    def test_endpoint_informa_modelo_no_entrenado(self, dynamic_model_path):
        dynamic_model_path.exists.return_value = False
        response = self.client.post(
            '/sequence_predict/',
            data=json.dumps({'frames': [[[0.0, 0.0, 0.0]] * 21]}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 503)

    def test_conserva_trayectoria_y_normaliza_ventana(self):
        sequence = np.zeros((8, 21, 3), dtype='float32')
        sequence[:, 0, 0] = np.linspace(0.0, 1.0, 8)
        sequence[:, 1, 1] = 1.0

        features, mask = build_sequence_features(sequence)

        self.assertEqual(features.shape, (32, 69))
        self.assertEqual(mask.shape, (32,))
        self.assertTrue(np.any(np.abs(features[:, 63:66]) > 0.0))
        self.assertTrue(np.any(np.abs(features[:, 66:69]) > 0.0))

    def test_rechaza_forma_de_landmarks_invalida(self):
        with self.assertRaises(ValueError):
            build_sequence_features(np.zeros((10, 20, 3), dtype='float32'))
