import pytest
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator, check_digit, normalize


@pytest.mark.parametrize('text', ['CSQU3054383', 'csqu 305438 3', 'CSQU-305438-3', 'ID: CSQU\n305438\n3'])
def test_valid_container(text):
    value = ContainerValidator().validate(OCRRead(text, .94))
    assert value.raw_text == text
    assert value.normalized_text == 'CSQU3054383'
    assert value.valid_format and value.valid_check_digit


def test_six_digit_prefix_derives_check_digit():
    val = ContainerValidator().validate(OCRRead('GXYU 509070', .95))
    assert val.normalized_text == 'GXYU5090704'
    assert val.valid_format and val.valid_check_digit and val.validation_status == 'VALID'


@pytest.mark.parametrize('text,status', [('CSQU3054384', 'INVALID_CHECK_DIGIT'), ('CSQA3054383', 'INVALID_FORMAT'),
                                       ('CSQU3O54383', 'INVALID_FORMAT'), ('CSQU30543833', 'INVALID_FORMAT'),
                                       ('No number visible', 'INVALID_FORMAT'), ('', 'INVALID_FORMAT')])
def test_invalid_ocr_is_not_trusted(text, status):
    result = ContainerValidator().validate(OCRRead(text, .99))
    assert result.validation_status == status
    assert not result.valid_check_digit


def test_ambiguous_ids_and_checksum_algorithm():
    assert check_digit('CSQU305438') == 3
    assert check_digit('MSCU663987') == 0
    result = ContainerValidator().validate(OCRRead('CSQU3054383 MSCU6639870', .9))
    assert result.validation_status == 'AMBIGUOUS'
    assert normalize(' abcu-123456-7 ') == 'ABCU1234567'
    with pytest.raises(ValueError):
        check_digit('INVALID')
