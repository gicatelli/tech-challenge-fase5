# OWASP Top 10 for LLM Applications — Mapeamento

## Referência
[OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## Mapeamento de Ameaças (7/10 mapeadas, 5+ com mitigação implementada)

| # | Ameaça OWASP | Status | Mitigação | Código |
|---|---|---|---|---|
| LLM01 | Prompt Injection | **MITIGADO** | 9 regex patterns no InputGuardrail | `src/security/guardrails.py:34-44` |
| LLM02 | Insecure Output Handling | **MITIGADO** | OutputGuardrail + Presidio PII removal | `src/security/guardrails.py:100-135` |
| LLM04 | Model Denial of Service | **MITIGADO** | max_length=4096 + Pydantic validation | `src/serving/app.py:72` |
| LLM06 | Sensitive Information Disclosure | **MITIGADO** | Presidio anonymizer + exfiltration patterns | `src/security/pii_detection.py` |
| LLM07 | Insecure Plugin Design | **MITIGADO** | Tools read-only, sem acesso a sistema | `src/agent/tools.py` |
| LLM08 | Excessive Agency | **MITIGADO** | max_iterations=10, tools com escopo fixo | `src/agent/react_agent.py:88` |
| LLM09 | Overreliance | **PARCIAL** | Disclaimer em respostas de previsão | `src/agent/tools.py:91` |

## Detalhamento

### LLM01 — Prompt Injection
- **Ameaça**: Atacante injeta instruções maliciosas no input
- **Mitigação**: 9 padrões regex detectam injections antes de chegar ao LLM
- **Evidência**: Red Team RT-01, RT-03, RT-07 — todos bloqueados
- **Código**: `src/security/guardrails.py` (INJECTION_PATTERNS)

### LLM02 — Insecure Output Handling
- **Ameaça**: LLM pode gerar output com PII ou código malicioso
- **Mitigação**: OutputGuardrail sanitiza PII (CPF, email, telefone, nomes)
- **Evidência**: Guardrail test — 3/3 PII detectados e removidos
- **Código**: `src/security/guardrails.py` (OutputGuardrail.sanitize)

### LLM04 — Model Denial of Service
- **Ameaça**: Input gigante causa consumo excessivo de recursos
- **Mitigação**: Limite de 4096 chars no Pydantic + guardrail
- **Evidência**: Red Team RT-06 — input de 5000 chars bloqueado
- **Código**: `src/serving/app.py` (QueryRequest.max_length)

### LLM06 — Sensitive Information Disclosure
- **Ameaça**: Modelo revela dados sensíveis do treinamento
- **Mitigação**: Presidio detecta e anonimiza PII; padrões de exfiltração bloqueiam queries
- **Evidência**: Red Team RT-02, RT-04 — exfiltração bloqueada, PII sanitizado
- **Código**: `src/security/pii_detection.py`, `src/security/guardrails.py`

### LLM07 — Insecure Plugin Design
- **Ameaça**: Tools executam ações perigosas (escrita, deleção, comandos)
- **Mitigação**: Todas as 4 tools são read-only (análise de dados, busca, cálculos)
- **Evidência**: Red Team RT-05 — "execute rm -rf" bloqueado
- **Código**: `src/agent/tools.py` (nenhuma tool executa comandos de sistema)

### LLM08 — Excessive Agency
- **Ameaça**: Agente executa ações além do necessário
- **Mitigação**: max_iterations=10, handle_parsing_errors=True
- **Evidência**: Agente para após resposta final ou 10 iterações
- **Código**: `src/agent/react_agent.py`

### LLM09 — Overreliance
- **Ameaça**: Usuário confia cegamente nas previsões do modelo
- **Mitigação**: Disclaimer "Previsões são estimativas e não garantem resultados futuros"
- **Status**: Parcial — pode ser melhorado com confidence intervals
- **Código**: `src/agent/tools.py:91` (campo "aviso" na resposta)
