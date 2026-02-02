if __name__ == "__main__":
    akb = AKBClient("http://localhost:8000")
    slm = SLMRunnerHF(...)
    inf = RecursiveInferencer(akb, slm)

    summary = inf.run(
        "신규 결제 플로우 장애 원인 분석 후 재발 방지 대책 수립", dummy_executor
    )
    print(summary["ok"])
    pass
