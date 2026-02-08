# 修复SRT解析错误计划

## 问题分析
错误信息：`'datetime.time' object has no attribute 'total_seconds'`

**根本原因**：在 `srt_service.py` 的 `parse_srt` 方法中，使用了 `sub.start.to_time()`，该方法返回 `datetime.time` 对象（表示一天中的时间），而不是 `datetime.timedelta` 对象（表示时间间隔）。`datetime.time` 对象没有 `total_seconds()` 方法。

**错误代码位置**：
```python
# 第24-26行，srt_service.py
"start": self._timedelta_to_seconds(sub.start.to_time()),
"end": self._timedelta_to_seconds(sub.end.to_time()),
"duration": self._timedelta_to_seconds(sub.duration.to_time()),
```

## 修复方案

### 方案1：使用pysrt的ordinal属性（推荐）
`pysrt.SubRipTime` 对象有 `ordinal` 属性，返回以毫秒为单位的时间戳。修复方法：

```python
def parse_srt(self, srt_path: str) -> List[Dict[str, Any]]:
    try:
        subs = pysrt.open(srt_path)
        subtitles = []
        
        for sub in subs:
            # 使用ordinal属性获取毫秒数，转换为秒
            start_seconds = sub.start.ordinal / 1000.0
            end_seconds = sub.end.ordinal / 1000.0
            duration_seconds = sub.duration.ordinal / 1000.0 if hasattr(sub.duration, 'ordinal') else end_seconds - start_seconds
            
            subtitle = {
                "index": sub.index,
                "start": start_seconds,
                "end": end_seconds,
                "duration": duration_seconds,
                "text": sub.text.strip(),
                "raw_start": str(sub.start),
                "raw_end": str(sub.end)
            }
            subtitles.append(subtitle)
        
        return subtitles
        
    except Exception as e:
        raise Exception(f"SRT解析失败: {str(e)}")
```

### 方案2：使用pysrt的时间组件计算
如果`ordinal`属性不可用，可以使用小时、分钟、秒、毫秒组件计算：

```python
def _subrip_time_to_seconds(self, subrip_time):
    """将SubRipTime对象转换为秒"""
    total_ms = (subrip_time.hours * 3600 + 
                subrip_time.minutes * 60 + 
                subrip_time.seconds) * 1000 + subrip_time.milliseconds
    return total_ms / 1000.0
```

### 方案3：修改_timedelta_to_seconds函数
修改`_timedelta_to_seconds`函数以同时支持`datetime.time`和`timedelta`对象：

```python
def _timedelta_to_seconds(self, td):
    """将timedelta或datetime.time对象转换为秒"""
    if hasattr(td, 'total_seconds'):  # timedelta对象
        return td.total_seconds()
    elif hasattr(td, 'hour'):  # datetime.time对象
        # 计算从午夜开始的秒数
        return (td.hour * 3600 + td.minute * 60 + td.second + td.microsecond / 1_000_000)
    elif hasattr(td, 'ordinal'):  # SubRipTime对象
        return td.ordinal / 1000.0
    else:
        raise ValueError(f"不支持的时间类型: {type(td)}")
```

## 具体实施步骤

### 第一步：修复parse_srt方法
1. 修改 `srt_service.py` 中的 `parse_srt` 方法
2. 使用 `sub.start.ordinal` 和 `sub.end.ordinal` 获取毫秒时间戳
3. 转换为秒：`秒数 = ordinal / 1000.0`

### 第二步：修复相关方法
1. 更新 `get_total_duration` 方法中的类似问题
2. 可选：添加辅助函数 `_subrip_time_to_seconds` 提高代码可读性

### 第三步：测试修复
1. 创建测试SRT文件进行验证
2. 测试完整的视频生成流程
3. 确保时间计算准确无误

## 代码修改细节

### 需要修改的文件
- `app/services/srt_service.py`

### 主要修改内容
1. **删除或修改 `_timedelta_to_seconds` 函数**：因为不再需要将`datetime.time`转换为`timedelta`
2. **直接计算秒数**：使用`ordinal`属性或时间组件计算
3. **保持向后兼容性**：确保其他代码（如`merge_consecutive_subtitles`）仍然能正常工作

## 测试验证

修复后需要验证：
1. ✅ SRT文件能够正常解析
2. ✅ 时间计算准确（开始时间、结束时间、持续时间）
3. ✅ 语义对齐模块能正确使用时间数据
4. ✅ 完整的视频生成流程正常工作

## 风险评估

1. **pysrt版本兼容性**：确保`ordinal`属性在所有pysrt版本中都可用
2. **时间精度**：毫秒转换为秒时保持足够精度
3. **异常处理**：添加适当的错误处理和类型检查

## 备选方案

如果`ordinal`属性不可用，使用时间组件计算：
```python
def _subrip_time_to_seconds(self, subrip_time):
    """将SubRipTime对象转换为秒"""
    return (subrip_time.hours * 3600 + 
            subrip_time.minutes * 60 + 
            subrip_time.seconds + 
            subrip_time.milliseconds / 1000.0)
```

## 时间估计
- 代码修改：15分钟
- 测试验证：15分钟
- 总计：约30分钟

---

确认此计划后，我将开始实施修复。