- Sequence Diagram Mermaid Code: 
    sequenceDiagram
    autonumber
    actor U as Kullanıcı
    participant O as Orchestrator
    participant S as Supervisor
    participant D as Data Analyst
    participant V as Visualization
    participant I as Insight
    participant E as Evaluation
    participant T as TraceLogger

    U->>O: soru + veri seti
    O->>S: classify_intent(soru)
    S-->>O: IntentResult (llm | rule_fallback)
    O->>T: log(IntentResult)
    O->>S: select_workflow(intent)
    S-->>O: WorkflowPlan (insight_focus)
    O->>T: log(WorkflowPlan)

    O->>D: run_data_analysis(şema, soru, workflow)
    Note over D: LLM planı → kolon doğrulama<br/>→ güvenli çalıştırma → gerçek istatistikler
    D-->>O: TransformPlan (summary_stats) | StepError
    O->>T: log(TransformPlan)

    O->>V: run_visualization(plan, veri)
    Note over V: izinli listeden seçim<br/>+ veriye bakan guardrail'ler
    V-->>O: ChartDecision (guardrails_applied)
    O->>T: log(ChartDecision)

    O->>I: run_insight(soru, summary_stats)
    Note over I: LLM cümlesi → sayı doğrulama<br/>→ gerekirse şablon yedeği
    I-->>O: InsightResult (source)
    O->>T: log(InsightResult)

    O->>E: run_evaluation(tüm çıktılar)
    Note over E: 8 kural, LLM yok
    E-->>O: EvalVerdict (passed, issues, warnings)
    O->>T: log(EvalVerdict)

    alt verdict FAILED
        O->>O: _blame() → suçlu ajan
        O->>D: hedefli yeniden koşum
        O->>E: yeniden değerlendirme
        E-->>O: EvalVerdict (retried_step, retry_helped)
        O->>T: log(EvalVerdict #2)
    end

    O-->>U: grafik + insight (+ varsa uyarı) + Agent Trace



- Flowchart Mermaid Code: 
    flowchart TD
    Q([Kullanıcı sorusu]) --> SUP[Supervisor<br/>niyet sınıflandırma]
    SUP -->|LLM geçerli| WF[İş akışı seçimi<br/>insight_focus]
    SUP -.->|LLM geçersiz| RULE[Kural yedeği<br/>anahtar kelime eşleşmesi] -.-> WF
    WF --> DA[Data Analyst<br/>dönüşüm planı + çalıştırma<br/>+ gerçek istatistikler]
    DA -->|StepError| STOP([Zincir durur<br/>yapılandırılmış hata + trace])
    DA --> VIZ[Visualization<br/>izinli listeden seçim]
    VIZ --> GUARD{Guardrail<br/>denetimi}
    GUARD -->|ihlal| FIX[Düzelt + kaydet<br/>guardrails_applied] --> INS
    GUARD -->|temiz| INS[Insight<br/>istatistiklerden cümle]
    INS --> VER{Groundedness<br/>doğrulayıcı}
    VER -->|sayı uydurma| TPL[Şablon yedeği<br/>source=template_fallback] --> EVAL
    VER -->|temiz| EVAL[Evaluation<br/>7 kural tabanlı kontrol]
    EVAL -->|PASSED| OUT([Cevap: grafik + insight<br/>+ Agent Trace])
    EVAL -->|FAILED| BLAME[Suçlu adımı belirle<br/>_blame haritası]
    BLAME -->|bir kez| RETRY[Hedefli retry:<br/>suçlu ajan + reddedilme gerekçesi<br/>prompt'a geri beslenir]
    RETRY --> EVAL2{Yeniden<br/>değerlendirme}
    EVAL2 -->|PASSED| OUT
    EVAL2 -->|hâlâ FAILED| WARN([Cevap + uyarı bayrağı<br/>şeffaf kusurlu teslim])