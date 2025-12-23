"""
Voice Control - Controle por Voz para USG Flow
===============================================
Sistema de reconhecimento de voz para operacao hands-free.
Usa Google Speech Recognition (requer internet) ou offline com Sphinx.

Comandos suportados:
- "freeze" / "congelar" - pausa/resume imagem
- "record" / "gravar" - inicia/para gravacao
- "screenshot" / "foto" - captura screenshot
- "next" / "proximo" - proxima janela (FAST) ou bloqueio (NERVE)
- "previous" / "anterior" - janela/bloqueio anterior
- "confirm" / "confirmar" - confirma janela FAST
- "reset" - reinicia exame FAST
"""

import speech_recognition as sr
import threading
import queue
import time
import subprocess
import logging
from typing import Callable, Dict, Optional, List

logger = logging.getLogger(__name__)


class VoiceController:
    """Controlador de voz para comandos hands-free."""

    # Mapeamento de comandos (portugues e ingles)
    COMMAND_ALIASES = {
        # Freeze/Pause
        'freeze': 'freeze',
        'congelar': 'freeze',
        'pausar': 'freeze',
        'pause': 'freeze',
        'parar': 'freeze',

        # Record
        'record': 'record',
        'gravar': 'record',
        'gravacao': 'record',
        'recording': 'record',

        # Screenshot
        'screenshot': 'screenshot',
        'foto': 'screenshot',
        'captura': 'screenshot',
        'capture': 'screenshot',
        'print': 'screenshot',

        # Next
        'next': 'next',
        'proximo': 'next',
        'proxima': 'next',
        'avancar': 'next',
        'forward': 'next',

        # Previous
        'previous': 'previous',
        'anterior': 'previous',
        'voltar': 'previous',
        'back': 'previous',

        # Confirm
        'confirm': 'confirm',
        'confirmar': 'confirm',
        'ok': 'confirm',
        'check': 'confirm',
        'done': 'confirm',

        # Reset
        'reset': 'reset',
        'reiniciar': 'reset',
        'restart': 'reset',
        'novo': 'reset',
        'new': 'reset',
    }

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Inicializa o controlador de voz.

        Args:
            callback: Funcao chamada quando um comando e reconhecido.
                      Recebe o nome do comando normalizado como argumento.
        """
        self.callback = callback
        self.recognizer = sr.Recognizer()
        self.microphone = None

        # Estado
        self.running = False
        self.listening = False
        self.enabled = False
        self._thread: Optional[threading.Thread] = None
        self._command_queue = queue.Queue()

        # Configuracoes
        self.energy_threshold = 300  # Sensibilidade do microfone
        self.pause_threshold = 0.5   # Segundos de silencio para parar escuta
        self.phrase_time_limit = 3   # Tempo maximo para uma frase

        # Historico de comandos
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 1.5  # Segundos entre comandos repetidos

        # Status para UI
        self.status = "IDLE"  # IDLE, LISTENING, PROCESSING, ERROR
        self.last_error = None

        # Inicializar microfone
        self._init_microphone()

    def _init_microphone(self):
        """Inicializa o microfone."""
        try:
            self.microphone = sr.Microphone()
            # Ajustar para ruido ambiente
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("Microfone inicializado")
        except Exception as e:
            logger.error(f"Erro ao inicializar microfone: {e}")
            self.last_error = str(e)
            self.microphone = None

    def _play_sound(self, sound_name: str):
        """Reproduz um som de feedback."""
        sounds = {
            'listening': '/System/Library/Sounds/Tink.aiff',
            'recognized': '/System/Library/Sounds/Pop.aiff',
            'error': '/System/Library/Sounds/Basso.aiff',
        }
        sound_file = sounds.get(sound_name)
        if sound_file:
            try:
                subprocess.Popen(
                    ['afplay', sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

    def _normalize_command(self, text: str) -> Optional[str]:
        """
        Normaliza o texto reconhecido para um comando padrao.

        Args:
            text: Texto reconhecido pelo speech recognition.

        Returns:
            Nome do comando normalizado ou None se nao reconhecido.
        """
        if not text:
            return None

        # Converter para minusculas e remover acentos comuns
        text = text.lower().strip()

        # Procurar comando no texto
        words = text.split()
        for word in words:
            # Remover acentos simples
            word = word.replace('á', 'a').replace('é', 'e').replace('í', 'i')
            word = word.replace('ó', 'o').replace('ú', 'u').replace('ã', 'a')
            word = word.replace('õ', 'o').replace('ç', 'c')

            if word in self.COMMAND_ALIASES:
                return self.COMMAND_ALIASES[word]

        return None

    def _listen_loop(self):
        """Loop principal de escuta (roda em thread separada)."""
        logger.info("Voice control thread iniciada")

        while self.running:
            if not self.enabled or self.microphone is None:
                time.sleep(0.1)
                continue

            try:
                self.status = "LISTENING"
                self.listening = True

                with self.microphone as source:
                    # Escutar audio
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=2,
                            phrase_time_limit=self.phrase_time_limit
                        )
                    except sr.WaitTimeoutError:
                        continue

                self.status = "PROCESSING"
                self.listening = False

                # Tentar reconhecer com Google (requer internet)
                try:
                    text = self.recognizer.recognize_google(audio, language='pt-BR')
                    logger.debug(f"Reconhecido: {text}")

                    command = self._normalize_command(text)
                    if command:
                        # Verificar cooldown
                        now = time.time()
                        if command == self.last_command and (now - self.last_command_time) < self.command_cooldown:
                            continue

                        self.last_command = command
                        self.last_command_time = now

                        # Feedback sonoro
                        self._play_sound('recognized')

                        # Notificar callback
                        if self.callback:
                            self.callback(command)
                        else:
                            self._command_queue.put(command)

                        logger.info(f"Comando reconhecido: {command}")

                except sr.UnknownValueError:
                    # Audio nao reconhecido
                    pass
                except sr.RequestError as e:
                    logger.warning(f"Erro no servico de reconhecimento: {e}")
                    self.last_error = "Sem conexao com internet"
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Erro no loop de escuta: {e}")
                self.status = "ERROR"
                self.last_error = str(e)
                time.sleep(1)

        self.status = "IDLE"
        self.listening = False
        logger.info("Voice control thread encerrada")

    def start(self):
        """Inicia o controlador de voz em background."""
        if self.running:
            return

        if self.microphone is None:
            logger.error("Nao foi possivel iniciar: microfone nao disponivel")
            return

        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Voice control iniciado")

    def stop(self):
        """Para o controlador de voz."""
        self.running = False
        self.enabled = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Voice control parado")

    def enable(self):
        """Ativa a escuta de comandos."""
        if self.microphone is None:
            logger.warning("Microfone nao disponivel")
            return False

        self.enabled = True
        self._play_sound('listening')
        logger.info("Voice control ativado")
        return True

    def disable(self):
        """Desativa a escuta de comandos."""
        self.enabled = False
        self.status = "IDLE"
        logger.info("Voice control desativado")

    def toggle(self) -> bool:
        """Alterna entre ativado/desativado. Retorna novo estado."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    def get_command(self, block: bool = False, timeout: float = None) -> Optional[str]:
        """
        Obtem o proximo comando da fila (se nao usando callback).

        Args:
            block: Se True, bloqueia ate um comando estar disponivel.
            timeout: Timeout em segundos (apenas se block=True).

        Returns:
            Nome do comando ou None.
        """
        try:
            return self._command_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self) -> Dict:
        """Retorna status atual para exibicao na UI."""
        return {
            'running': self.running,
            'enabled': self.enabled,
            'listening': self.listening,
            'status': self.status,
            'last_command': self.last_command,
            'last_error': self.last_error,
            'microphone_ok': self.microphone is not None,
        }


# Teste standalone
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    def on_command(cmd):
        print(f"\n>>> COMANDO: {cmd} <<<\n")

    vc = VoiceController(callback=on_command)
    vc.start()
    vc.enable()

    print("Voice Control ativo. Fale um comando (freeze, record, screenshot, next, confirm)...")
    print("Pressione Ctrl+C para sair.\n")

    try:
        while True:
            status = vc.get_status()
            print(f"\rStatus: {status['status']:<12} | Ultimo: {status['last_command'] or '-':<10}", end='', flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEncerrando...")
        vc.stop()
