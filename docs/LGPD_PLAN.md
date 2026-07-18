# Plano de Conformidade LGPD — Datathon Fase 05

## Referência Legal

Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais (LGPD)

## Mapeamento de Dados Pessoais

### Dados Coletados/Processados

| Dado | Categoria LGPD | Base Legal | Finalidade | Retenção |
|------|---------------|------------|------------|----------|
| [Campo 1] | Dado pessoal | Legítimo interesse | [Finalidade] | [Período] |
| [Campo 2] | Dado sensível | Consentimento | [Finalidade] | [Período] |
| [Campo 3] | Dado pessoal | Execução contratual | [Finalidade] | [Período] |

### Fluxo de Dados

```
[Fonte] → [Ingestão] → [Processamento] → [Armazenamento] → [Uso] → [Descarte]
```

## Princípios LGPD Aplicados

### Art. 6º — Princípios

| Princípio | Implementação |
|-----------|---------------|
| Finalidade | Dados usados exclusivamente para [objetivo do sistema] |
| Adequação | Apenas dados necessários são coletados |
| Necessidade | Minimização: apenas features relevantes |
| Livre acesso | API permite consulta de dados processados |
| Qualidade | Validação de schema + testes de integridade |
| Transparência | System Card documenta uso de dados |
| Segurança | Criptografia, guardrails, PII detection |
| Prevenção | Monitoramento contínuo, drift detection |
| Não discriminação | Análise de fairness documentada |
| Responsabilização | Audit trail via MLflow |

## Medidas Técnicas Implementadas

### 1. Anonimização e Pseudonimização
- **Ferramenta**: Microsoft Presidio
- **Entidades**: CPF, CNPJ, Nome, Email, Telefone, Cartão
- **Aplicação**: Output do LLM é sanitizado antes de retornar ao usuário
- **Código**: `src/security/pii_detection.py`

### 2. Minimização de Dados
- Apenas features necessárias para o modelo são processadas
- Dados brutos não são armazenados após processamento
- DVC versiona dados sem expor conteúdo no Git

### 3. Segurança
- Secrets em `.env` (nunca hardcoded)
- HTTPS para comunicação
- Autenticação na API (produção)
- Logging de acessos

### 4. Direitos do Titular
- **Acesso**: API permite consulta
- **Correção**: Pipeline permite reprocessamento
- **Eliminação**: Dados podem ser removidos do vector store
- **Portabilidade**: Export em formato padrão (JSON/CSV)

## Relatório de Impacto (RIPD)

### Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Vazamento de PII via LLM | Média | Alto | OutputGuardrail + Presidio |
| Reidentificação | Baixa | Alto | Anonimização robusta |
| Uso indevido de dados | Baixa | Médio | Controle de acesso |
| Viés discriminatório | Média | Alto | Análise de fairness |

### Medidas de Mitigação
1. Guardrails de input e output
2. Detecção automática de PII
3. Monitoramento contínuo
4. Audit trail completo
5. Treinamento da equipe

## Encarregado (DPO)

- **Responsável**: [Nome do responsável no grupo]
- **Contato**: [Email]

## Revisão

Este plano deve ser revisado:
- A cada nova versão do modelo
- Quando novos dados forem incorporados
- Quando houver mudança na legislação
- Periodicamente (mínimo trimestral)
