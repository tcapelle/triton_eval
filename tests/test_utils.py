import pytest
import asyncio
from datasets import Dataset
from triton_eval.utils import map, compare_outputs


class TestMapFunction:
    """Test suite for the async map function."""

    def test_map_with_list_of_dicts_new_columns(self):
        """Test map function with list of dicts, function returns new columns."""
        async def run_test():
            # Test data
            data = [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30},
                {"id": 3, "name": "Charlie", "age": 35}
            ]
            
            # Function that adds new columns
            def add_info(row):
                return {
                    "name_upper": row["name"].upper(),
                    "age_category": "young" if row["age"] < 30 else "adult"
                }
            
            # Execute
            result = await map(data, add_info, num_proc=2)
            
            # Verify
            assert len(result) == 3
            
            # Check first row
            assert result[0]["id"] == 1
            assert result[0]["name"] == "Alice"
            assert result[0]["age"] == 25
            assert result[0]["name_upper"] == "ALICE"
            assert result[0]["age_category"] == "young"
            
            # Check second row
            assert result[1]["id"] == 2
            assert result[1]["name"] == "Bob"
            assert result[1]["age"] == 30
            assert result[1]["name_upper"] == "BOB"
            assert result[1]["age_category"] == "adult"
        
        asyncio.run(run_test())

    def test_map_with_list_of_dicts_full_row_update(self):
        """Test map function with list of dicts, function returns full updated row."""
        async def run_test():
            # Test data
            data = [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30}
            ]
            
            # Function that returns full updated row
            def transform_row(row):
                return {
                    "id": row["id"],
                    "value": row["value"] * 2,
                    "doubled": True,
                    "original_value": row["value"]
                }
            
            # Execute
            result = await map(data, transform_row, num_proc=1)
            
            # Verify
            assert len(result) == 3
            
            # Check that original data is preserved and new data is added
            assert result[0]["id"] == 1
            assert result[0]["value"] == 20  # Updated value
            assert result[0]["doubled"] == True  # New field
            assert result[0]["original_value"] == 10  # New field
        
        asyncio.run(run_test())

    def test_map_with_hf_dataset_new_columns(self):
        """Test map function with HuggingFace Dataset, function returns new columns."""
        async def run_test():
            # Create HF Dataset
            data = {
                "text": ["hello world", "foo bar", "test message"],
                "label": [0, 1, 0]
            }
            dataset = Dataset.from_dict(data)
            
            # Function that adds new columns
            def process_text(row):
                return {
                    "word_count": len(row["text"].split()),
                    "text_length": len(row["text"]),
                    "is_positive": row["label"] == 1
                }
            
            # Execute
            result = await map(dataset, process_text, num_proc=2)
            
            # Verify
            assert len(result) == 3
            
            # Check first row
            assert result[0]["text"] == "hello world"
            assert result[0]["label"] == 0
            assert result[0]["word_count"] == 2
            assert result[0]["text_length"] == 11
            assert result[0]["is_positive"] == False
            
            # Check second row (positive label)
            assert result[1]["text"] == "foo bar"
            assert result[1]["label"] == 1
            assert result[1]["is_positive"] == True
        
        asyncio.run(run_test())

    def test_map_with_hf_dataset_full_row_update(self):
        """Test map function with HuggingFace Dataset, function returns full updated row."""
        async def run_test():
            # Create HF Dataset
            data = {
                "input": ["a", "b", "c"],
                "target": [1, 2, 3]
            }
            dataset = Dataset.from_dict(data)
            
            # Function that returns full updated row
            def transform_data(row):
                return {
                    "input": row["input"].upper(),
                    "target": row["target"] * 10,
                    "processed": True,
                    "original_target": row["target"]
                }
            
            # Execute
            result = await map(dataset, transform_data, num_proc=1)
            
            # Verify
            assert len(result) == 3
            assert result[0]["input"] == "A"
            assert result[0]["target"] == 10
            assert result[0]["processed"] == True
            assert result[0]["original_target"] == 1
        
        asyncio.run(run_test())

    def test_map_empty_dataset(self):
        """Test map function with empty dataset."""
        async def run_test():
            data = []
            
            def dummy_func(row):
                return {"processed": True}
            
            result = await map(data, dummy_func, num_proc=1)
            
            assert result == []
        
        asyncio.run(run_test())

    def test_map_single_item(self):
        """Test map function with single item."""
        async def run_test():
            data = [{"id": 1, "value": "test"}]
            
            def add_length(row):
                return {"length": len(row["value"])}
            
            result = await map(data, add_length, num_proc=1)
            
            assert len(result) == 1
            assert result[0]["id"] == 1
            assert result[0]["value"] == "test"
            assert result[0]["length"] == 4
        
        asyncio.run(run_test())

    def test_map_preserves_original_data(self):
        """Test that map function preserves original data when adding new columns."""
        async def run_test():
            original_data = [
                {"name": "Alice", "age": 25, "city": "NYC"},
                {"name": "Bob", "age": 30, "city": "LA"}
            ]
            
            def add_info(row):
                return {"name_length": len(row["name"])}
            
            result = await map(original_data, add_info, num_proc=1)
            
            # Verify original data is preserved
            for i, row in enumerate(result):
                assert row["name"] == original_data[i]["name"]
                assert row["age"] == original_data[i]["age"]
                assert row["city"] == original_data[i]["city"]
                assert "name_length" in row
        
        asyncio.run(run_test())

    def test_map_overwrites_existing_keys(self):
        """Test that map function overwrites existing keys when function returns them."""
        async def run_test():
            data = [{"value": 10, "status": "old"}]
            
            def update_row(row):
                return {
                    "value": row["value"] * 2,
                    "status": "updated",
                    "new_field": "added"
                }
            
            result = await map(data, update_row, num_proc=1)
            
            assert len(result) == 1
            assert result[0]["value"] == 20  # Overwritten
            assert result[0]["status"] == "updated"  # Overwritten
            assert result[0]["new_field"] == "added"  # New
        
        asyncio.run(run_test())

    def test_map_with_different_num_proc(self):
        """Test map function with different num_proc values."""
        async def run_test():
            data = [{"id": i, "value": i * 2} for i in range(10)]
            
            def square_value(row):
                return {"squared": row["value"] ** 2}
            
            # Test with different concurrency levels
            for num_proc in [1, 3, 5]:
                result = await map(data, square_value, num_proc=num_proc)
                
                assert len(result) == 10
                for i, row in enumerate(result):
                    assert row["id"] == i
                    assert row["value"] == i * 2
                    assert row["squared"] == (i * 2) ** 2
        
        asyncio.run(run_test())

    def test_map_function_returns_empty_dict(self):
        """Test map function when processing function returns empty dict."""
        async def run_test():
            data = [{"id": 1, "name": "test"}]
            
            def empty_func(row):
                return {}
            
            result = await map(data, empty_func, num_proc=1)
            
            assert len(result) == 1
            # Original data should be preserved
            assert result[0]["id"] == 1
            assert result[0]["name"] == "test"
        
        asyncio.run(run_test())

    def test_map_function_returns_none_values(self):
        """Test map function when processing function returns None values."""
        async def run_test():
            data = [{"id": 1, "value": "test"}]
            
            def add_none_values(row):
                return {
                    "none_field": None,
                    "valid_field": "valid"
                }
            
            result = await map(data, add_none_values, num_proc=1)
            
            assert len(result) == 1
            assert result[0]["id"] == 1
            assert result[0]["value"] == "test"
            assert result[0]["none_field"] is None
            assert result[0]["valid_field"] == "valid"
        
        asyncio.run(run_test())

    def test_map_with_complex_data_types(self):
        """Test map function with complex data types (lists, dicts, etc.)."""
        async def run_test():
            data = [
                {
                    "id": 1,
                    "tags": ["python", "async"],
                    "metadata": {"created": "2024", "author": "test"}
                }
            ]
            
            def process_complex(row):
                return {
                    "tag_count": len(row["tags"]),
                    "has_python_tag": "python" in row["tags"],
                    "author_upper": row["metadata"]["author"].upper(),
                    "new_metadata": {"processed": True}
                }
            
            result = await map(data, process_complex, num_proc=1)
            
            assert len(result) == 1
            row = result[0]
            
            # Original data preserved
            assert row["id"] == 1
            assert row["tags"] == ["python", "async"]
            assert row["metadata"] == {"created": "2024", "author": "test"}
            
            # New data added
            assert row["tag_count"] == 2
            assert row["has_python_tag"] == True
            assert row["author_upper"] == "TEST"
            assert row["new_metadata"] == {"processed": True}
        
        asyncio.run(run_test())

    def test_map_progress_printing(self, capsys):
        """Test that map function prints progress information."""
        async def run_test():
            data = [{"id": i} for i in range(3)]
            
            def simple_func(row):
                return {"processed": True}
            
            await map(data, simple_func, num_proc=1)
        
        asyncio.run(run_test())
        
        # Check that progress was printed
        captured = capsys.readouterr()
        assert "Completed 1 / 3" in captured.out
        assert "Completed 2 / 3" in captured.out
        assert "Completed 3 / 3" in captured.out

    def test_map_function_signature(self):
        """Test that map function has the expected signature."""
        import inspect
        
        sig = inspect.signature(map)
        params = list(sig.parameters.keys())
        
        assert params == ["ds", "func", "num_proc"]
        assert sig.parameters["num_proc"].default == 10
        
        # Check that it's an async function
        assert asyncio.iscoroutinefunction(map)


class TestCompareOutputs:
    """Test suite for the compare_outputs function."""

    def test_identical_outputs_pass(self, capsys):
        """Test that identical tensor outputs pass comparison."""
        expected_str = """{'test_case_1': tensor([1., 2., 3.], device='cuda:0'), 'test_case_2': tensor([4, 5, 6], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1., 2., 3.], device='cuda:0'), 'test_case_2': tensor([4, 5, 6], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is True
        assert len(result["results"]) == 2
        
        # Check that all test cases passed
        statuses = [r[1] for r in result["results"]]
        assert all(status == 'PASS' for status in statuses)
        
        # Check printed output
        captured = capsys.readouterr()
        assert "test_case_1: PASS" in captured.out
        assert "test_case_2: PASS" in captured.out

    def test_floating_point_tolerance(self, capsys):
        """Test floating point comparison with tolerance."""
        expected_str = """{'test_case_1': tensor([1.0000, 2.0000], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1.0000001, 1.9999999], device='cuda:0')}"""
        
        # Should pass with default tolerance
        result = compare_outputs(expected_str, actual_str)
        assert result["match"] is True
        
        # Should fail with very strict tolerance
        result = compare_outputs(expected_str, actual_str, rtol=1e-10, atol=1e-10)
        assert result["match"] is False
        assert result["results"][0][1] == 'FAIL'

    def test_shape_mismatch(self, capsys):
        """Test that shape mismatches are detected."""
        expected_str = """{'test_case_1': tensor([1., 2.], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1., 2., 3.], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        assert result["results"][0][1] == 'SHAPE_MISMATCH'
        assert "(2,) vs (3,)" in result["results"][0][2]
        
        captured = capsys.readouterr()
        assert "test_case_1: SHAPE_MISMATCH" in captured.out

    def test_missing_test_case_in_actual(self, capsys):
        """Test detection of missing test cases in actual output."""
        expected_str = """{'test_case_1': tensor([1., 2.], device='cuda:0'), 'test_case_2': tensor([3., 4.], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1., 2.], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        
        # Find the missing case result
        missing_result = None
        for name, status, msg, err in result["results"]:
            if name == 'test_case_2':
                missing_result = (name, status, msg, err)
                break
        
        assert missing_result is not None
        assert missing_result[1] == 'MISSING_IN_ACTUAL'
        
        captured = capsys.readouterr()
        assert "test_case_2: MISSING_IN_ACTUAL" in captured.out

    def test_unexpected_test_case_in_actual(self, capsys):
        """Test detection of unexpected test cases in actual output."""
        expected_str = """{'test_case_1': tensor([1., 2.], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1., 2.], device='cuda:0'), 'test_case_2': tensor([3., 4.], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        
        # Find the unexpected case result
        unexpected_result = None
        for name, status, msg, err in result["results"]:
            if name == 'test_case_2':
                unexpected_result = (name, status, msg, err)
                break
        
        assert unexpected_result is not None
        assert unexpected_result[1] == 'UNEXPECTED_IN_ACTUAL'
        
        captured = capsys.readouterr()
        assert "test_case_2: UNEXPECTED_IN_ACTUAL" in captured.out

    def test_integer_comparison(self, capsys):
        """Test exact comparison for integer tensors."""
        expected_str = """{'test_case_1': tensor([1, 2, 3], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1, 2, 4], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        assert result["results"][0][1] == 'FAIL'
        assert result["results"][0][2] == 'values_differ'
        
        captured = capsys.readouterr()
        assert "test_case_1: FAIL (values_differ)" in captured.out

    def test_mixed_data_types(self, capsys):
        """Test comparison with mixed floating point and integer tensors."""
        expected_str = """{'float_test': tensor([1.5, 2.5], device='cuda:0'), 'int_test': tensor([1, 2], device='cuda:0')}"""
        actual_str = """{'float_test': tensor([1.5, 2.5], device='cuda:0'), 'int_test': tensor([1, 2], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is True
        assert len(result["results"]) == 2
        
        # Both should pass
        statuses = [r[1] for r in result["results"]]
        assert all(status == 'PASS' for status in statuses)

    def test_complex_tensor_format(self, capsys):
        """Test with complex multi-line tensor format like in the user's example."""
        expected_str = """{'test_case_4': tensor([2.1614, 2.3551, 1.0104, 2.2787, 1.1415, 1.3815, 2.4606, 2.1596, 1.2988,
        2.5350, 1.9523, 2.0083, 1.4691, 2.3788, 2.5429, 1.7166, 2.4421, 3.0356,
        1.6107, 1.8944, 1.9710, 2.4425, 2.1108, 1.5378, 3.0331, 1.2209, 2.4659,
        2.7442, 2.2449, 1.4087], device='cuda:0')}"""
        
        actual_str = """{'test_case_4': tensor([2.1614, 2.3551, 1.0104, 2.2787, 1.1415, 1.3815, 2.4606, 2.1596, 1.2988,
        2.5350, 1.9523, 2.0083, 1.4691, 2.3788, 2.5429, 1.7166, 2.4421, 3.0356,
        1.6107, 1.8944, 1.9710, 2.4425, 2.1108, 1.5378, 3.0331, 1.2209, 2.4659,
        2.7442, 2.2449, 1.4087], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is True
        assert result["results"][0][1] == 'PASS'

    def test_tensor_with_dtype_annotation(self, capsys):
        """Test tensors with basic format."""
        expected_str = """{'test_case_1': tensor([1, 2], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1, 2], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is True
        assert result["results"][0][1] == 'PASS'

    def test_realistic_mismatch_scenario(self, capsys):
        """Test a realistic scenario similar to user's example with mismatches."""
        expected_str = """{'test_case_1': tensor([3., 5.], device='cuda:0'), 'test_case_2': tensor([7., 9.], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([3., 3.], device='cuda:0'), 'test_case_2': tensor([7., 7., 7.], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        
        # Check results
        results_dict = {name: (status, msg) for name, status, msg, _ in result["results"]}
        
        # test_case_1 should fail due to value mismatch
        assert results_dict['test_case_1'][0] == 'FAIL'
        
        # test_case_2 should fail due to shape mismatch
        assert results_dict['test_case_2'][0] == 'SHAPE_MISMATCH'
        assert "(2,) vs (3,)" in results_dict['test_case_2'][1]

    def test_parse_error_handling(self, capsys):
        """Test handling of parse errors in tensor strings."""
        expected_str = """{'test_case_1': tensor([1., 2.], device='cuda:0')}"""
        # Malformed tensor string
        actual_str = """{'test_case_1': tensor([1., 2., device='cuda:0')}"""  # Missing closing bracket
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        assert result["results"][0][1] == 'PARSE_ERROR'
        
        captured = capsys.readouterr()
        assert "test_case_1: PARSE_ERROR" in captured.out

    def test_empty_strings(self):
        """Test behavior with empty input strings."""
        result = compare_outputs("", "")
        
        assert result["match"] is True
        assert result["results"] == []

    def test_custom_tolerance_values(self, capsys):
        """Test custom rtol and atol values."""
        expected_str = """{'test_case_1': tensor([1.0, 2.0], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1.01, 2.01], device='cuda:0')}"""
        
        # Should fail with strict tolerance
        result = compare_outputs(expected_str, actual_str, rtol=1e-3, atol=1e-3)
        assert result["match"] is False
        
        # Should pass with loose tolerance
        result = compare_outputs(expected_str, actual_str, rtol=1e-1, atol=1e-1)
        assert result["match"] is True

    def test_max_error_reporting(self, capsys):
        """Test that max error is correctly reported for floating point comparisons."""
        expected_str = """{'test_case_1': tensor([1.0, 2.0], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([1.1, 2.05], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str, rtol=1e-10, atol=1e-10)
        
        assert result["match"] is False
        
        # Check that max_error is reported
        max_error = result["results"][0][3]
        assert max_error is not None
        assert max_error > 0.05  # Should be around 0.1 (max difference)
        
        # Check that it's in the message
        assert "max_error=" in result["results"][0][2]

    def test_user_example_format(self, capsys):
        """Test with the actual format from user's example."""
        expected_str = """{'test_case_1': tensor([3., 5.], device='cuda:0'), 'test_case_2': tensor([7., 9.], device='cuda:0'), 'test_case_3': tensor([5., 2.], device='cuda:0')}"""
        actual_str = """{'test_case_1': tensor([3., 3.], device='cuda:0'), 'test_case_2': tensor([7., 7., 7.], device='cuda:0'), 'test_case_3': tensor([5., 5.], device='cuda:0')}"""
        
        result = compare_outputs(expected_str, actual_str)
        
        assert result["match"] is False
        assert len(result["results"]) == 3
        
        # Check specific results
        results_dict = {name: (status, msg) for name, status, msg, _ in result["results"]}
        
        # test_case_1: value mismatch (3.,5.) vs (3.,3.)
        assert results_dict['test_case_1'][0] == 'FAIL'
        
        # test_case_2: shape mismatch (2,) vs (3,)
        assert results_dict['test_case_2'][0] == 'SHAPE_MISMATCH'
        
        # test_case_3: value mismatch (5.,2.) vs (5.,5.)
        assert results_dict['test_case_3'][0] == 'FAIL'


# Additional integration-style tests
class TestMapIntegration:
    """Integration tests for the map function with more realistic scenarios."""

    def test_text_processing_pipeline(self):
        """Test a realistic text processing pipeline."""
        async def run_test():
            # Simulate a text processing dataset
            data = [
                {"text": "Hello World!", "category": "greeting"},
                {"text": "Python is awesome", "category": "programming"},
                {"text": "Machine Learning rocks", "category": "ai"}
            ]
            
            def text_processor(row):
                text = row["text"]
                return {
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "has_exclamation": "!" in text,
                    "text_lower": text.lower(),
                    "category_upper": row["category"].upper()
                }
            
            result = await map(data, text_processor, num_proc=2)
            
            assert len(result) == 3
            
            # Check first item
            assert result[0]["text"] == "Hello World!"
            assert result[0]["word_count"] == 2
            assert result[0]["char_count"] == 12
            assert result[0]["has_exclamation"] == True
            assert result[0]["text_lower"] == "hello world!"
            assert result[0]["category_upper"] == "GREETING"
        
        asyncio.run(run_test())

    def test_numerical_computation_pipeline(self):
        """Test a numerical computation pipeline."""
        async def run_test():
            # Simulate numerical data
            data = [
                {"values": [1, 2, 3, 4, 5], "multiplier": 2},
                {"values": [10, 20, 30], "multiplier": 3},
                {"values": [100], "multiplier": 1}
            ]
            
            def numerical_processor(row):
                values = row["values"]
                multiplier = row["multiplier"]
                
                processed_values = [v * multiplier for v in values]
                
                return {
                    "processed_values": processed_values,
                    "sum_original": sum(values),
                    "sum_processed": sum(processed_values),
                    "count": len(values),
                    "avg_original": sum(values) / len(values),
                    "max_processed": max(processed_values)
                }
            
            result = await map(data, numerical_processor, num_proc=1)
            
            assert len(result) == 3
            
            # Check first item
            assert result[0]["processed_values"] == [2, 4, 6, 8, 10]
            assert result[0]["sum_original"] == 15
            assert result[0]["sum_processed"] == 30
            assert result[0]["count"] == 5
            assert result[0]["avg_original"] == 3.0
            assert result[0]["max_processed"] == 10
        
        asyncio.run(run_test())

    def test_with_huggingface_dataset_realistic(self):
        """Test with a more realistic HuggingFace dataset scenario."""
        async def run_test():
            # Create a dataset similar to what might be used in practice
            data = {
                "input_text": [
                    "Translate to French: Hello",
                    "Translate to French: Goodbye", 
                    "Translate to French: Thank you"
                ],
                "target_text": [
                    "Bonjour",
                    "Au revoir",
                    "Merci"
                ],
                "difficulty": ["easy", "easy", "medium"]
            }
            dataset = Dataset.from_dict(data)
            
            def create_training_example(row):
                return {
                    "prompt": f"Task: {row['input_text']}\nResponse: {row['target_text']}",
                    "input_length": len(row["input_text"]),
                    "target_length": len(row["target_text"]),
                    "is_difficult": row["difficulty"] == "hard",
                    "language_pair": "en-fr"
                }
            
            result = await map(dataset, create_training_example, num_proc=2)
            
            assert len(result) == 3
            
            # Check that original data is preserved
            assert all("input_text" in row for row in result)
            assert all("target_text" in row for row in result)
            assert all("difficulty" in row for row in result)
            
            # Check new fields
            assert all("prompt" in row for row in result)
            assert all("language_pair" in row for row in result)
            
            # Check specific example
            first_result = result[0]
            expected_prompt = "Task: Translate to French: Hello\nResponse: Bonjour"
            assert first_result["prompt"] == expected_prompt
            assert first_result["input_length"] == len("Translate to French: Hello")
            assert first_result["target_length"] == len("Bonjour")
            assert first_result["is_difficult"] == False
            assert first_result["language_pair"] == "en-fr"
        
        asyncio.run(run_test()) 