def test_source_selector_app_2():
    from code_similarity_tool.code_similarity import CodeSimilarityTool
    from code_similarity_tool.selection import SourceSelectorApp

    tool = CodeSimilarityTool()
    app = SourceSelectorApp(tool)

    assert app.tool is tool