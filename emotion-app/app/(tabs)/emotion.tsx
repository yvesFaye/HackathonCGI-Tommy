import { Audio } from 'expo-av';
import Constants from 'expo-constants';
import React, { useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Platform, Pressable, StyleSheet, Text, View } from 'react-native';

type EmotionResponse = {
  file?: string;
  emotion?: string;
  score?: number;
  model_mode?: string;
};

// Mets ici l'IP Wi-Fi du PC (la meme que celle testee avec /health).
const MANUAL_API_BASE = 'https://poor-spoons-push.loca.lt';
const FALLBACK_API_BASE = MANUAL_API_BASE;

function getApiBaseUrl(): string {
  if (MANUAL_API_BASE) {
    return MANUAL_API_BASE;
  }

  const hostUri = Constants.expoConfig?.hostUri;
  if (!hostUri) {
    return FALLBACK_API_BASE;
  }

  const host = hostUri.split(':')[0];
  if (!host) {
    return FALLBACK_API_BASE;
  }

  return `http://${host}:8000`;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

function formatNetworkError(stage: string, url: string): string {
  return `Timeout reseau vers API (${stage}). URL=${url}. Verifie: meme Wi-Fi, firewall Windows port 8000, serveur Python actif.`;
}

export default function EmotionScreen() {
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);
  const analyzeUrl = `${apiBaseUrl}/analyze-audio`;
  const healthUrl = `${apiBaseUrl}/health`;

  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [lastResult, setLastResult] = useState<EmotionResponse | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const isBusy = useMemo(() => isRecording || isSending, [isRecording, isSending]);

  const recordAndSend = async () => {
    setLastError(null);
    setLastResult(null);

    const permission = await Audio.requestPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission micro', 'Autorise le microphone pour tester.');
      return;
    }

    let recording: Audio.Recording | null = null;
    try {
      setIsRecording(true);
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      recording = new Audio.Recording();
      await recording.prepareToRecordAsync({
        android: {
          extension: '.wav',
          outputFormat: Audio.AndroidOutputFormat.DEFAULT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 128000,
        },
        ios: {
          extension: '.wav',
          audioQuality: Audio.IOSAudioQuality.HIGH,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 128000,
          linearPCMBitDepth: 16,
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {},
      });
      await recording.startAsync();

      // Enregistre environ 3 secondes.
      await new Promise((resolve) => setTimeout(resolve, 3000));

      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();

      if (!uri) {
        throw new Error('Aucun fichier audio produit par Expo.');
      }

      setIsRecording(false);
      setIsSending(true);

      const form = new FormData();
      form.append('file', {
        uri,
        name: Platform.OS === 'ios' || Platform.OS === 'android' ? 'sample.wav' : 'sample.webm',
        type: Platform.OS === 'ios' || Platform.OS === 'android' ? 'audio/wav' : 'audio/webm',
      } as never);

      // Soft preflight: if this fails we still try the upload once.
      try {
        await fetchWithTimeout(
          healthUrl,
          {
            method: 'GET',
          },
          15000,
        );
      } catch {
        // Non-blocking by design.
      }

      const response = await fetchWithTimeout(
        analyzeUrl,
        {
          method: 'POST',
          body: form,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        },
        180000,
      );

      const data = (await response.json()) as EmotionResponse | { detail?: string };
      if (!response.ok) {
        const detail = (data as { detail?: string }).detail || 'Erreur API inconnue.';
        throw new Error(detail);
      }

      setLastResult(data as EmotionResponse);
    } catch (error) {
      let message = error instanceof Error ? error.message : 'Erreur inconnue.';
      if (message.includes('Network request failed')) {
        message = formatNetworkError('connexion', analyzeUrl);
      }
      if (message.includes('aborted') || message.includes('Aborted')) {
        message = formatNetworkError('timeout', analyzeUrl);
      }
      setLastError(message);
    } finally {
      setIsRecording(false);
      setIsSending(false);

      try {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
        });
      } catch {
        // Ignore audio mode reset errors.
      }
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Tommy the helper backend</Text>
      <Text style={styles.subtitle}>Maintiens le serveur Python allume sur le port 8000.</Text>
      <Text style={styles.subtitle}>API: {analyzeUrl}</Text>

      <Pressable style={[styles.button, isBusy && styles.buttonDisabled]} onPress={recordAndSend} disabled={isBusy}>
        <Text style={styles.buttonText}>
          {isRecording ? 'Enregistrement...' : isSending ? 'Envoi...' : 'Enregistrer et envoyer'}
        </Text>
      </Pressable>

      {isBusy ? <ActivityIndicator size="small" color="#0a7ea4" /> : null}

      {lastResult ? (
        <View style={styles.card}>
          <Text style={styles.row}>emotion: {lastResult.emotion ?? '-'}</Text>
          <Text style={styles.row}>score: {lastResult.score ?? '-'}</Text>
          <Text style={styles.row}>modele: {lastResult.model_mode ?? '-'}</Text>
          <Text style={styles.row}>file: {lastResult.file ?? '-'}</Text>
        </View>
      ) : null}

      {lastError ? <Text style={styles.error}>Erreur: {lastError}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    gap: 14,
    backgroundColor: '#f4f8fa',
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#0e2431',
  },
  subtitle: {
    fontSize: 14,
    color: '#48606c',
  },
  button: {
    backgroundColor: '#0a7ea4',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  card: {
    borderRadius: 12,
    backgroundColor: '#ffffff',
    padding: 14,
    borderWidth: 1,
    borderColor: '#d6e3ea',
    gap: 4,
  },
  row: {
    fontSize: 14,
    color: '#1e3642',
  },
  error: {
    color: '#b42318',
    fontSize: 14,
  },
});
