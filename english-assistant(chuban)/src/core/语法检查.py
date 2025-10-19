from language_tool_python import LanguageTool

def translate_error_message(rule_id, message):
    """将常见的英文错误描述翻译为中文"""
    # 常见错误类型的中文翻译
    rule_translations = {
        'AI_HYDRA_LEO_CPT_ARE_IS': '动词“are”与主语不一致，单数主语应使用“is”',
        'MORFOLOGIK_RULE_EN_US': '可能的拼写错误',
        'UPPERCASE_SENTENCE_START': '句子开头应该大写',
        'WHITESPACE_RULE': '多余的空格',
        'COMMA_PARENTHESIS_WHITESPACE': '逗号和括号前后的空格问题',
        'A_INFINITIVE': '不定式动词前的冠词使用错误',
        'EN_A_VS_AN': '冠词 a/an 使用错误',
        'HE_VERB_AGR': '人称代词与动词不一致',
        'BEEN_PART_AGREEMENT': '助动词与过去分词不一致',
        'SENTENCE_FRAGMENT': '句子不完整',
        'DOUBLE_PUNCTUATION': '重复的标点符号'
    }
    
    # 关键词匹配翻译
    keyword_translations = {
        "doesn't seem to fit": "似乎不符合语法规则",
        "more formally correct": "更正式的表达方式",
        "agreement error": "主谓一致性错误",
        "spelling mistake": "拼写错误",
        "possible typo": "可能的打字错误",
        "wrong word": "错误的单词",
        "missing word": "缺少单词",
        "extra word": "多余的单词",
        "word order": "词序错误",
        "punctuation": "标点符号错误",
        "capitalization": "大小写错误"
    }
    
    # 首先检查规则ID的精确匹配
    if rule_id in rule_translations:
        return rule_translations[rule_id]
    
    # 如果没有精确匹配，尝试关键词匹配
    message_lower = message.lower()
    for english_keyword, chinese_translation in keyword_translations.items():
        if english_keyword in message_lower:
            return chinese_translation
    
    # 如果都没有匹配，返回原始消息
    return message

def analyze_grammar(text):
    """使用LanguageTool进行语法分析"""
    tool = None
    try:
        # 使用 LanguageTool 的公共 API
        tool = LanguageTool('en-US', remote_server='https://api.languagetool.org')
        matches = tool.check(text)
        
        # 获取修正后的文本
        try:
            corrected_text = tool.correct(text)
        except Exception as e:
            print(f"获取修正文本失败: {e}")
            corrected_text = text

        if not matches:
            return {
                "status": "success", 
                "message": "语法正确！",
                "corrected_text": corrected_text
            }

        report = {
            "error_count": len(matches),
            "errors": [],
            "corrected_text": corrected_text
        }

        for match in matches:
            # 翻译错误消息
            translated_message = translate_error_message(match.ruleId, match.message)
            
            error_info = {
                "rule_id": match.ruleId,
                "message": translated_message,  # 使用翻译后的消息
                "original_message": match.message,  # 保留原始消息作为参考
                "context": match.context,
                "replacements": [str(r) for r in match.replacements]  # 确保是字符串列表
            }
            
            # 添加错误位置信息
            if hasattr(match, 'offset') and hasattr(match, 'errorLength'):
                error_info["offset"] = match.offset
                error_info["length"] = match.errorLength
                error_info["error_text"] = text[match.offset:match.offset + match.errorLength]
            
            report["errors"].append(error_info)

        return report
        
    except Exception as e:
        print(f"语法检查失败: {str(e)}")
        return {
            "status": "error",
            "message": f"语法检查服务不可用: {str(e)}",
            "error_count": 0,
            "errors": [],
            "corrected_text": text
        }
    finally:
        # 确保资源被正确释放
        if tool is not None:
            try:
                tool.close()
            except Exception as close_error:
                print(f"关闭LanguageTool时出错: {close_error}")