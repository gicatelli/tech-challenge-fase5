# OWASP Top 10 for LLM Applications — Mapeamento de Ameaças

Referência: [OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## Resumo

| # | Ameaça | Risco | Mitigação | Status |
|---|--------|-------|-----------|--------|
| 1 | Prompt Injection | Alto | InputGuardrail | ✅ Implementado |
| 2 | Insecure Output Handling | Alto | OutputGuardrail + PII | ✅ Implementado |
| 3 | Training Data Poisoning | Médio | Validação de dados | ✅ Implementado |
| 4 | Model Denial of Service | Médio | Rate limiting + timeout | ✅ Implementado |
| 5 | Supply Chain Vulnerabilities | Médio | Bandit + deps pinadas | ✅ Implementado |
| 6 | Sensitive Information Disclosure | Alto | Presidio PII | ✅ Implementado |
| 7 | Insecure Plugin Design | Baixo | Tool validation | ⚠️ Parcial |
| 8 | Excessive Agency | Médio | max_iterations + approval | ✅ Implementado |
| 9 | Overreliance | Médio | Disclaimers + human-in-loop | ✅ Implementado |
| 10 | Model Theft | Baixo | API auth + rate limit | ⚠️ Parcial |

## Detalhamento

### LLM01: Prompt Injection

**Descrição**: Atacante manipula o LLM via inputs crafted para ignorar instruções.

**Risco para o sistema**: ALTO — O agente tem acesso a tools que podem executar ações.

**Mitigações implementadas**:
1. `InputGuardrail` com regex patterns para detectar injection
2. Limite de tamanho de input (4096 chars)
3. Sanitização de caracteres especiais
4. Logging de tentativas para auditoria

**Código**: `src/security/guardrails.py` → `InputGuardrail.validate()`

---

### LLM02: Insecure Output Handling

**Descrição**: Output do LLM é usado sem validação, podendo conter código malicioso ou PII.

**Risco para o sistema**: ALTO — Output pode vazar dados sensíveis.

**Mitigações implementadas**:
1. `OutputGuardrail` com detecção de PII via Presidio
2. Validação de padrões de sistema no output
3. Sanitização antes de retornar ao usuário

**Código**: `src/security/guardrails.py` → `OutputGuardrail.sanitize()`

---

### LLM03: Training Data Poisoning

**Descrição**: Dados de treinamento contaminados afetam comportamento do modelo.

**Risco para o sistema**: MÉDIO — Dados vêm de fonte controlada (empresa).

**Mitigações implementadas**:
1. Validação de schema com Pandera
2. Testes de integridade de dados
3. Versionamento com DVC (auditabilidade)
4. Monitoramento de drift

---

### LLM04: Model Denial of Service

**Descrição**: Atacante sobrecarrega o sistema com queries complexas.

**Risco para o sistema**: MÉDIO — API pública pode ser alvo.

**Mitigações implementadas**:
1. `max_iterations=10` no agente (limita loops)
2. Timeout em requests
3. Limite de tamanho de input
4. Health check para detectar degradação

---

### LLM05: Supply Chain Vulnerabilities

**Descrição**: Dependências maliciosas ou vulneráveis.

**Risco para o sistema**: MÉDIO — Muitas dependências Python.

**Mitigações implementadas**:
1. Bandit scan no CI/CD
2. Dependências com versões pinadas no pyproject.toml
3. Pre-commit hooks para segurança
4. Scan de vulnerabilidades

---

### LLM06: Sensitive Information Disclosure

**Descrição**: LLM vaza informações sensíveis (PII, secrets, dados internos).

**Risco para o sistema**: ALTO — Sistema processa dados financeiros.

**Mitigações implementadas**:
1. Presidio para detecção de PII (CPF, CNPJ, email, telefone)
2. Anonimização automática no output
3. `.env` para secrets (nunca hardcoded)
4. `.gitignore` para dados sensíveis

**Código**: `src/security/pii_detection.py`

---

### LLM07: Insecure Plugin Design

**Descrição**: Tools/plugins do agente sem validação adequada.

**Risco para o sistema**: BAIXO — Tools são read-only no MVP.

**Mitigações implementadas**:
1. Tools com escopo limitado (read-only)
2. Validação de input em cada tool
3. Logging de todas as execuções

---

### LLM08: Excessive Agency

**Descrição**: LLM tem autonomia excessiva para executar ações.

**Risco para o sistema**: MÉDIO — Agente pode executar tools.

**Mitigações implementadas**:
1. `max_iterations=10` (limita ações)
2. Tools com escopo restrito
3. Logging de todos os steps
4. Human-in-the-loop para ações críticas

---

### LLM09: Overreliance

**Descrição**: Usuários confiam cegamente nas respostas do LLM.

**Risco para o sistema**: MÉDIO — Domínio financeiro requer precisão.

**Mitigações implementadas**:
1. Disclaimer nas respostas
2. Citação de fontes (RAG)
3. Indicação de confiança
4. Documentação de limitações

---

### LLM10: Model Theft

**Descrição**: Extração do modelo via API.

**Risco para o sistema**: BAIXO — Usa API OpenAI (modelo não é local).

**Mitigações implementadas**:
1. Rate limiting na API
2. Autenticação (quando em produção)
3. Logging de acessos
