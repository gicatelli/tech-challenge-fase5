# Plano de Conformidade LGPD — Datathon Fase 05

## 1. Mapeamento de Dados

### Dados coletados pelo sistema

| Dado | Tipo | Pessoal? | Base Legal |
|------|------|:--------:|------------|
| Preços históricos PETR4/VALE3/ITUB4 | Financeiro público | Não | Legítimo interesse |
| Volume de negociação | Financeiro público | Não | Legítimo interesse |
| Queries dos usuários | Texto livre | **Pode conter PII** | Consentimento |
| Respostas do agente | Texto gerado | Não (sanitizado) | N/A |
| Logs de telemetria | Operacional | Pode conter PII indireta | Legítimo interesse |

### Classificação
- **Dados financeiros**: públicos, disponíveis no Yahoo Finance. Não configuram dados pessoais.
- **Queries dos usuários**: podem conter nomes, CPFs ou informações pessoais acidentalmente.
- **Outputs do LLM**: sanitizados pelo Presidio antes de retornar ao usuário.

## 2. Princípios LGPD Aplicados

| Princípio | Aplicação no sistema |
|-----------|---------------------|
| **Finalidade** | Dados usados exclusivamente para previsão de preços e análise de risco |
| **Adequação** | Apenas dados de mercado público necessários são coletados |
| **Necessidade** | Mínimo de dados: OHLCV de 3 ações, sem dados pessoais |
| **Transparência** | System Card público, explicabilidade documentada |
| **Segurança** | Guardrails, PII removal, logs anonimizados |
| **Não discriminação** | Análise de fairness (sem viés direcional detectado) |

## 3. Medidas Técnicas Implementadas

### 3.1 Proteção de PII no Input
- InputGuardrail bloqueia tentativas de exfiltração de dados pessoais
- Padrões regex detectam queries pedindo CPF, senhas, dados bancários
- **Código**: `src/security/guardrails.py`

### 3.2 Proteção de PII no Output
- Presidio Analyzer detecta entidades: PERSON, EMAIL, PHONE, BR_CPF, BR_CNPJ
- Presidio Anonymizer substitui por placeholders: `<PESSOA>`, `<CPF>`, `<EMAIL>`
- **Código**: `src/security/pii_detection.py`

### 3.3 Logs Anonimizados
- Telemetria trunca outputs a 200 chars
- Queries logadas sem dados pessoais
- Arquivo local (não transmitido a terceiros)
- **Código**: `src/monitoring/telemetry.py`

### 3.4 Secrets Management
- API keys em `.env` (nunca commitado)
- `.env.example` como template sem valores reais
- Nenhum secret hardcoded no código

## 4. RIPD Simplificado (Relatório de Impacto)

### 4.1 Descrição do Tratamento
- **Operação**: Coleta de dados públicos de mercado + processamento por LLM
- **Volume**: ~6000 registros financeiros (3 ações × 2000 dias)
- **Dados pessoais**: Nenhum coletado intencionalmente. PII detectado em queries é bloqueado/anonimizado.

### 4.2 Necessidade e Proporcionalidade
- Dados de mercado são públicos e essenciais para previsão de preços
- Nenhuma coleta de dados pessoais é necessária para o funcionamento
- Alternativa a dados pessoais: não aplicável (sistema não usa dados pessoais)

### 4.3 Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:---:|:---:|-----------|
| PII em queries do usuário | Média | Baixo | InputGuardrail + OutputGuardrail |
| PII em logs | Baixa | Baixo | Truncamento + anonimização |
| Vazamento de API keys | Baixa | Alto | .env + .gitignore |
| Viés em recomendações | Baixa | Médio | Disclaimer + análise fairness |

### 4.4 Medidas de Mitigação
- [x] PII detection e anonymization implementados
- [x] Guardrails de input e output ativos
- [x] Secrets management via .env
- [x] Logs truncados e sem PII
- [x] Análise de fairness documentada

## 5. Direitos do Titular

| Direito | Aplicabilidade | Implementação |
|---------|:-:|---------------|
| Acesso | Baixa (não coletamos PII) | N/A |
| Correção | Baixa | N/A |
| Eliminação | Média (logs) | Logs podem ser limpos manualmente |
| Portabilidade | Baixa | N/A |
| Oposição | Média | Usuário pode parar de usar o sistema |

## 6. Conclusão

O sistema apresenta **risco baixo** sob a LGPD porque:
1. Não coleta dados pessoais intencionalmente
2. Dados financeiros utilizados são públicos
3. PII acidental em queries é bloqueado/anonimizado
4. Não há compartilhamento com terceiros
5. Logs são truncados e armazenados localmente
