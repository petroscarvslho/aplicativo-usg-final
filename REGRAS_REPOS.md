# REGRAS PARA EVITAR CONFUSAO ENTRE REPOS

## OS 2 PROJETOS

| Projeto | Pasta | GitHub | Funcao |
|---------|-------|--------|--------|
| TRAINER | `~/ultrasound-needle-trainer` | petroscarvslho/ultrasound-needle-trainer | Treina modelos |
| APP | `~/aplicativo-usg-final` | petroscarvslho/aplicativo-usg-final | Usa modelos |

## REGRA 1: IDENTIFICAR O TERMINAL

Sempre que abrir um terminal, rode:

```bash
# Para TRAINER:
cd ~/ultrasound-needle-trainer && export PS1="🏋️ TRAINER $ "

# Para APP:
cd ~/aplicativo-usg-final && export PS1="📱 APP $ "
```

## REGRA 2: NUNCA VERSIONAR ARQUIVOS GRANDES

Arquivos que NUNCA devem ir pro git:
- *.pt (modelos PyTorch)
- *.pth (checkpoints)
- *.npy (arrays NumPy)
- *.h5 (modelos Keras)
- Pastas: processed/, raw/, exports/, checkpoints/

## REGRA 3: ANTES DE COMMITAR

```bash
pwd
git remote -v
git status
```

Se tiver arquivo grande, NAO commite!

## REGRA 4: UM TERMINAL POR PROJETO

- Terminal 1: SOMENTE para TRAINER
- Terminal 2: SOMENTE para APP

Lembre-se: Os repos sao INDEPENDENTES!
