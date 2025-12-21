"""
Premium Design System
Sistema de design profissional inspirado em Apple, Butterfly iQ3 e apps médicos premium
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import colorsys


@dataclass
class Color:
    """Representa uma cor no sistema de design"""
    r: int
    g: int
    b: int
    a: float = 1.0
    
    def to_hex(self) -> str:
        """Converte para hex"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_rgb(self) -> Tuple[int, int, int]:
        """Converte para tupla RGB"""
        return (self.r, self.g, self.b)

    def to_bgr(self) -> Tuple[int, int, int]:
        """Converte para tupla BGR (para OpenCV)"""
        return (self.b, self.g, self.r)

    def __getitem__(self, index: int):
        """Permite acessar como tupla: color[0], color[1], color[2]"""
        return (self.r, self.g, self.b, self.a)[index]
    
    def to_rgba(self) -> Tuple[int, int, int, float]:
        """Converte para tupla RGBA"""
        return (self.r, self.g, self.b, self.a)
    
    def with_alpha(self, alpha: float) -> 'Color':
        """Retorna nova cor com alpha diferente"""
        return Color(self.r, self.g, self.b, alpha)
    
    def lighten(self, amount: float = 0.1) -> 'Color':
        """Retorna versão mais clara da cor"""
        h, l, s = colorsys.rgb_to_hls(self.r/255, self.g/255, self.b/255)
        l = min(1.0, l + amount)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return Color(int(r*255), int(g*255), int(b*255), self.a)
    
    def darken(self, amount: float = 0.1) -> 'Color':
        """Retorna versão mais escura da cor"""
        h, l, s = colorsys.rgb_to_hls(self.r/255, self.g/255, self.b/255)
        l = max(0.0, l - amount)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return Color(int(r*255), int(g*255), int(b*255), self.a)


@dataclass
class Gradient:
    """Representa um gradiente"""
    start: Color
    end: Color
    angle: int = 0  # 0=horizontal, 90=vertical, etc.
    
    def to_css(self) -> str:
        """Converte para CSS gradient"""
        return f"linear-gradient({self.angle}deg, {self.start.to_hex()}, {self.end.to_hex()})"


@dataclass
class Shadow:
    """Representa uma sombra"""
    x: int
    y: int
    blur: int
    spread: int
    color: Color
    
    def to_css(self) -> str:
        """Converte para CSS box-shadow"""
        return f"{self.x}px {self.y}px {self.blur}px {self.spread}px {self.color.to_hex()}"


class PremiumDesignSystem:
    """
    Sistema de design premium completo
    Inspirado em Apple Human Interface Guidelines + Butterfly iQ3
    """
    
    def __init__(self, theme: str = 'dark'):
        """
        Inicializa sistema de design
        
        Args:
            theme: 'dark' ou 'light'
        """
        self.theme = theme
        self._init_colors()
        self._init_typography()
        self._init_spacing()
        self._init_shadows()
        self._init_animations()
    
    def _init_colors(self):
        """Inicializa paleta de cores"""
        if self.theme == 'dark':
            # Dark Theme - Premium
            self.colors = {
                # Backgrounds
                'bg_primary': Color(15, 15, 20),  # Quase preto
                'bg_secondary': Color(25, 25, 32),  # Cinza muito escuro
                'bg_tertiary': Color(35, 35, 45),  # Cinza escuro
                'bg_elevated': Color(45, 45, 58),  # Cinza médio-escuro
                'bg_glass': Color(255, 255, 255, 0.05),  # Glassmorphism
                
                # Surfaces
                'surface': Color(28, 28, 35),
                'surface_elevated': Color(38, 38, 48),
                'surface_glass': Color(255, 255, 255, 0.08),
                
                # Text
                'text_primary': Color(255, 255, 255),
                'text_secondary': Color(200, 200, 210),
                'text_tertiary': Color(150, 150, 165),
                'text_disabled': Color(100, 100, 115),
                
                # Accents
                'accent_primary': Color(0, 122, 255),  # Azul Apple
                'accent_secondary': Color(52, 199, 89),  # Verde
                'accent_tertiary': Color(255, 149, 0),  # Laranja
                
                # Butterfly Brand
                'butterfly_blue': Color(0, 180, 255),
                'butterfly_cyan': Color(0, 230, 255),
                
                # Medical Colors
                'medical_red': Color(255, 59, 48),  # Artéria
                'medical_blue': Color(10, 132, 255),  # Veia
                'medical_yellow': Color(255, 204, 0),  # Nervo
                'medical_green': Color(48, 209, 88),  # Agulha
                'medical_purple': Color(191, 90, 242),  # Músculo
                
                # Status
                'success': Color(52, 199, 89),
                'warning': Color(255, 204, 0),
                'error': Color(255, 59, 48),
                'info': Color(0, 122, 255),
                
                # Ultrasound specific
                'us_background': Color(0, 0, 0),  # Fundo B-Mode
                'us_tissue': Color(180, 180, 180),  # Tecido
                'doppler_red': Color(255, 50, 50),  # Afastando
                'doppler_blue': Color(50, 50, 255),  # Aproximando

                # Aliases para compatibilidade
                'text': Color(255, 255, 255),  # Alias para text_primary
                'border': Color(60, 60, 75),  # Cor de borda padrão
                'danger': Color(255, 59, 48),  # Alias para error
            }
        else:
            # Light Theme
            self.colors = {
                'bg_primary': Color(255, 255, 255),
                'bg_secondary': Color(248, 248, 250),
                'bg_tertiary': Color(240, 240, 245),
                'bg_elevated': Color(250, 250, 252),
                'bg_glass': Color(0, 0, 0, 0.05),
                
                'surface': Color(252, 252, 254),
                'surface_elevated': Color(255, 255, 255),
                'surface_glass': Color(0, 0, 0, 0.03),
                
                'text_primary': Color(0, 0, 0),
                'text_secondary': Color(60, 60, 67),
                'text_tertiary': Color(142, 142, 147),
                'text_disabled': Color(199, 199, 204),
                
                'accent_primary': Color(0, 122, 255),
                'accent_secondary': Color(52, 199, 89),
                'accent_tertiary': Color(255, 149, 0),
                
                'butterfly_blue': Color(0, 150, 230),
                'butterfly_cyan': Color(0, 200, 240),
                
                'medical_red': Color(255, 59, 48),
                'medical_blue': Color(0, 122, 255),
                'medical_yellow': Color(255, 204, 0),
                'medical_green': Color(52, 199, 89),
                'medical_purple': Color(175, 82, 222),
                
                'success': Color(52, 199, 89),
                'warning': Color(255, 149, 0),
                'error': Color(255, 59, 48),
                'info': Color(0, 122, 255),
                
                'us_background': Color(0, 0, 0),
                'us_tissue': Color(180, 180, 180),
                'doppler_red': Color(255, 50, 50),
                'doppler_blue': Color(50, 50, 255),

                # Aliases para compatibilidade
                'text': Color(0, 0, 0),  # Alias para text_primary
                'border': Color(200, 200, 210),  # Cor de borda padrão
                'danger': Color(255, 59, 48),  # Alias para error
            }

        # Gradients
        self.gradients = {
            'primary': Gradient(
                self.colors['accent_primary'],
                self.colors['accent_primary'].lighten(0.2),
                135
            ),
            'butterfly': Gradient(
                self.colors['butterfly_blue'],
                self.colors['butterfly_cyan'],
                135
            ),
            'medical': Gradient(
                self.colors['medical_blue'],
                self.colors['medical_green'],
                90
            ),
            'glass': Gradient(
                Color(255, 255, 255, 0.1),
                Color(255, 255, 255, 0.05),
                180
            )
        }
    
    def _init_typography(self):
        """Inicializa tipografia"""
        # Fontes (San Francisco no macOS, fallback para Segoe UI)
        self.fonts = {
            'display': ('SF Pro Display', 'Segoe UI', 'system-ui', 'sans-serif'),
            'text': ('SF Pro Text', 'Segoe UI', 'system-ui', 'sans-serif'),
            'mono': ('SF Mono', 'Consolas', 'Monaco', 'monospace'),
            'rounded': ('SF Pro Rounded', 'Segoe UI', 'system-ui', 'sans-serif'),
        }
        
        # Tamanhos (seguindo escala modular 1.25)
        self.font_sizes = {
            'xs': 11,
            'sm': 13,
            'base': 15,
            'md': 17,
            'lg': 20,
            'xl': 24,
            '2xl': 28,
            '3xl': 34,
            '4xl': 40,
            '5xl': 48,
            '6xl': 60,
        }
        
        # Pesos
        self.font_weights = {
            'light': 300,
            'regular': 400,
            'medium': 500,
            'semibold': 600,
            'bold': 700,
            'heavy': 800,
        }
        
        # Line heights
        self.line_heights = {
            'tight': 1.2,
            'normal': 1.5,
            'relaxed': 1.75,
        }
    
    def _init_spacing(self):
        """Inicializa espaçamento"""
        # Sistema de espaçamento 8pt grid
        base = 8
        self.spacing = {
            '0': 0,
            '1': base * 0.5,   # 4px
            '2': base * 1,     # 8px
            '3': base * 1.5,   # 12px
            '4': base * 2,     # 16px
            '5': base * 2.5,   # 20px
            '6': base * 3,     # 24px
            '8': base * 4,     # 32px
            '10': base * 5,    # 40px
            '12': base * 6,    # 48px
            '16': base * 8,    # 64px
            '20': base * 10,   # 80px
            '24': base * 12,   # 96px
        }
        
        # Border radius
        self.radius = {
            'none': 0,
            'sm': 4,
            'md': 8,
            'lg': 12,
            'xl': 16,
            '2xl': 24,
            'full': 9999,
        }
    
    def _init_shadows(self):
        """Inicializa sombras"""
        shadow_color = Color(0, 0, 0, 0.15) if self.theme == 'dark' else Color(0, 0, 0, 0.1)
        
        self.shadows = {
            'none': None,
            'sm': Shadow(0, 1, 2, 0, shadow_color),
            'md': Shadow(0, 4, 6, -1, shadow_color),
            'lg': Shadow(0, 10, 15, -3, shadow_color),
            'xl': Shadow(0, 20, 25, -5, shadow_color),
            '2xl': Shadow(0, 25, 50, -12, shadow_color),
        }
        
        # Glows (para elementos interativos)
        self.glows = {
            'primary': Shadow(0, 0, 20, 0, self.colors['accent_primary'].with_alpha(0.5)),
            'success': Shadow(0, 0, 20, 0, self.colors['success'].with_alpha(0.5)),
            'error': Shadow(0, 0, 20, 0, self.colors['error'].with_alpha(0.5)),
        }
    
    def _init_animations(self):
        """Inicializa configurações de animação"""
        # Durações (ms)
        self.durations = {
            'instant': 0,
            'fast': 150,
            'normal': 250,
            'slow': 350,
            'slower': 500,
        }
        
        # Easing functions
        self.easings = {
            'linear': 'linear',
            'ease': 'ease',
            'ease_in': 'ease-in',
            'ease_out': 'ease-out',
            'ease_in_out': 'ease-in-out',
            'spring': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',  # Bounce
            'smooth': 'cubic-bezier(0.4, 0.0, 0.2, 1)',  # Material
        }
    
    def get_color(self, name: str) -> Color:
        """Retorna cor por nome"""
        return self.colors.get(name, self.colors['text_primary'])
    
    def get_gradient(self, name: str) -> Gradient:
        """Retorna gradiente por nome"""
        return self.gradients.get(name, self.gradients['primary'])
    
    def get_shadow(self, name: str) -> Shadow:
        """Retorna sombra por nome"""
        return self.shadows.get(name)
    
    def get_glow(self, name: str) -> Shadow:
        """Retorna glow por nome"""
        return self.glows.get(name)
    
    def get_font(self, style: str = 'text') -> Tuple[str, ...]:
        """Retorna família de fontes"""
        return self.fonts.get(style, self.fonts['text'])
    
    def get_font_size(self, size: str = 'base') -> int:
        """Retorna tamanho de fonte"""
        return self.font_sizes.get(size, self.font_sizes['base'])
    
    def get_spacing(self, size: str = '4') -> int:
        """Retorna espaçamento"""
        return self.spacing.get(size, self.spacing['4'])
    
    def get_radius(self, size: str = 'md') -> int:
        """Retorna border radius"""
        return self.radius.get(size, self.radius['md'])
    
    def get_duration(self, speed: str = 'normal') -> int:
        """Retorna duração de animação"""
        return self.durations.get(speed, self.durations['normal'])
    
    def get_easing(self, style: str = 'smooth') -> str:
        """Retorna easing function"""
        return self.easings.get(style, self.easings['smooth'])
    
    def create_glassmorphism_style(self) -> Dict[str, Any]:
        """Cria estilo glassmorphism"""
        return {
            'background': self.colors['surface_glass'],
            'backdrop_blur': 20,
            'border': (1, self.colors['text_primary'].with_alpha(0.1)),
            'shadow': self.shadows['lg'],
        }
    
    def create_card_style(self, elevated: bool = False) -> Dict[str, Any]:
        """Cria estilo de card"""
        return {
            'background': self.colors['surface_elevated'] if elevated else self.colors['surface'],
            'radius': self.radius['xl'],
            'shadow': self.shadows['md'] if elevated else self.shadows['sm'],
            'padding': self.spacing['6'],
        }
    
    def create_button_style(self, variant: str = 'primary') -> Dict[str, Any]:
        """Cria estilo de botão"""
        styles = {
            'primary': {
                'background': self.colors['accent_primary'],
                'text': self.colors['text_primary'],
                'hover_background': self.colors['accent_primary'].lighten(0.1),
                'active_background': self.colors['accent_primary'].darken(0.1),
            },
            'secondary': {
                'background': self.colors['surface_elevated'],
                'text': self.colors['text_primary'],
                'hover_background': self.colors['surface_elevated'].lighten(0.05),
                'active_background': self.colors['surface_elevated'].darken(0.05),
            },
            'ghost': {
                'background': Color(0, 0, 0, 0),
                'text': self.colors['text_secondary'],
                'hover_background': self.colors['surface_glass'],
                'active_background': self.colors['surface_elevated'],
            },
            'danger': {
                'background': self.colors['error'],
                'text': self.colors['text_primary'],
                'hover_background': self.colors['error'].lighten(0.1),
                'active_background': self.colors['error'].darken(0.1),
            },
        }
        
        base_style = styles.get(variant, styles['primary'])
        base_style.update({
            'radius': self.radius['lg'],
            'padding': (self.spacing['3'], self.spacing['6']),
            'font_size': self.font_sizes['base'],
            'font_weight': self.font_weights['semibold'],
            'shadow': self.shadows['sm'],
            'duration': self.durations['fast'],
            'easing': self.easings['smooth'],
        })
        
        return base_style


# Instância global
_design_system: PremiumDesignSystem = None


def get_design_system(theme: str = 'dark') -> PremiumDesignSystem:
    """
    Retorna instância singleton do design system
    
    Args:
        theme: 'dark' ou 'light'
        
    Returns:
        PremiumDesignSystem
    """
    global _design_system
    if _design_system is None or _design_system.theme != theme:
        _design_system = PremiumDesignSystem(theme=theme)
    return _design_system
